# dataset_generator/mimo_dataset_generator.py
# Modular MIMO Dataset Generation Framework
# Generates scalable and reproducible MIMO communication datasets with support for multiple modulation schemes.
#
# Key Features:
# - Configurable dataset generation for diverse MIMO setups (e.g., 4x4, QPSK, 16QAM, 64QAM).
# - Integrates advanced channel modeling (e.g., Rayleigh fading) and path loss computations (FSPL).
# - Supports sample generation per modulation scheme based on defined simulation parameters.
# - Includes metrics calculation (e.g., SNR, spectral efficiency) for performance evaluation.
# - Validates and verifies dataset integrity at every stage of generation.
#
# Simplifications:
# - Focused on static user scenarios with fixed antenna configurations for reproducibility.
# - Excludes mobility models (e.g., Doppler effects) for simpler and faster dataset generation.
#
# Tensor Dimensionality:
# - Validates tensor shapes at every stage to ensure consistency.
# - Typical channel response shape: (Batch Size, Num RX Antennas, Num TX Antennas).
# - Ensures dimensionality consistency before reshaping or applying matrix operations.
#
# Updates:
# - Added per-modulation scheme sample generation with validation for `samples_per_modulation`.
# - Improved distance validation in channel generation to ensure realistic and physical plausibility.
# - Streamlined dataset generation for easy integration with reinforcement learning pipelines.
#
# Scope:
# - This module focuses on dataset generation for MIMO simulation tasks, supporting training and evaluation of machine learning models.
# - Real-time updates and advanced user mobility models are excluded for simplicity.

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from Data.utill.logging_config import configure_logging, LoggerManager
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import h5py
import numpy as np
import tensorflow as tf
import sionna as sn
from sionna.channel.tr38901 import AntennaArray
from config.system_parameters import SystemParameters
from ..utill.logging_config import MIMOLogger, LoggerManager
from core.metrics_calculator import MetricsCalculator
from config.system_parameters import SystemParameters
from integrity.dataset_integrity_checker import MIMODatasetIntegrityChecker
from utill.tensor_shape_validator import validate_mimo_tensor_shapes
from utill.tensor_shape_validator import validate_mimo_metrics
from tqdm import tqdm
from core.path_loss_model import PathLossManager
import sionna as sn

class MIMODatasetGenerator:
    def __init__(
        self, 
        system_params: SystemParameters = None,
        logger=None
    ):
        """Initialize MIMO dataset generator"""
        self.logger = logger or LoggerManager.get_logger(
            name='MIMODatasetGenerator', 
            log_level='INFO'
        )
        
        self.mimo_logger = MIMOLogger()
        self.system_params = system_params or SystemParameters()
        self.metrics_calculator = MetricsCalculator(self.system_params)
        
        # Initialize PathLossManager
        self.path_loss_manager = PathLossManager(self.system_params)
        
        # Initialize Sionna components
        self._setup_sionna_components()
        
        # Set validation thresholds
        self.validation_thresholds = {
            'path_loss': {'min': 20.0, 'max': 160.0},
            'sinr': {'min': -20.0, 'max': 30.0},
            'spectral_efficiency': {'min': 0.0, 'max': 40.0},
            'effective_snr': {'min': -10.0, 'max': 40.0},
            'channel_response': {'min': -100.0, 'max': 100.0}
        }
        
    def _setup_sionna_components(self):
        """Setup Sionna channel models and antenna arrays"""
        try:
            import sionna as sn
            from sionna.channel import RayleighBlockFading
            from sionna.mapping import Mapper
            from sionna.ofdm import ResourceGrid, LSChannelEstimator, PilotPattern
            import numpy as np
            import tensorflow as tf

            # Create pilot pattern first
            print("[DEBUG] Creating pilot mask...")
            pilot_mask = np.zeros([
                self.system_params.num_tx_antennas,
                self.system_params.num_streams,
                self.system_params.num_ofdm_symbols,
                self.system_params.num_subcarriers
            ], dtype=bool)

            # Set pilot positions with appropriate spacing based on simulation plan
            pilot_time_spacing = 4  # Based on coherence time
            pilot_freq_spacing = 4  # Based on coherence bandwidth
            
            for tx in range(self.system_params.num_tx_antennas):
                for stream in range(self.system_params.num_streams):
                    for t in range(0, self.system_params.num_ofdm_symbols, pilot_time_spacing):
                        for f in range(0, self.system_params.num_subcarriers, pilot_freq_spacing):
                            pilot_mask[tx, stream, t, f] = True

            num_pilots = np.sum(pilot_mask[0, 0])
            print(f"[DEBUG] Number of pilots per stream: {num_pilots}")

            # Generate QPSK pilot symbols
            print("[DEBUG] Generating pilot symbols...")
            qpsk_symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
            pilot_symbols = np.zeros([
                self.system_params.num_tx_antennas,
                self.system_params.num_streams,
                num_pilots
            ], dtype=np.complex64)

            # Fill pilot symbols with fixed random seed for reproducibility
            rng = np.random.default_rng(self.system_params.random_seed)
            for tx in range(self.system_params.num_tx_antennas):
                for stream in range(self.system_params.num_streams):
                    pilot_symbols[tx, stream] = rng.choice(qpsk_symbols, num_pilots)

            # Create pilot pattern
            print("[DEBUG] Creating pilot pattern...")
            pilot_pattern = PilotPattern(
                mask=tf.cast(pilot_mask, tf.bool),
                pilots=tf.cast(pilot_symbols, tf.complex64)
            )

            # Initialize ResourceGrid with pilot pattern
            print("[DEBUG] Initializing OFDM Resource Grid...")
            self.resource_grid = ResourceGrid(
                num_ofdm_symbols=self.system_params.num_ofdm_symbols,
                fft_size=self.system_params.num_subcarriers,
                subcarrier_spacing=self.system_params.subcarrier_spacing,
                num_tx=self.system_params.num_tx_antennas,
                num_streams_per_tx=self.system_params.num_streams,
                pilot_pattern=pilot_pattern,  # Pass the pilot pattern here
                cyclic_prefix_length=self.system_params.num_subcarriers // 4
            )

            # Initialize channel estimator with the resource grid
            print("[DEBUG] Initializing channel estimator...")
            self.channel_estimator = LSChannelEstimator(
                resource_grid=self.resource_grid,
                interpolation_type="lin" 
            )
            
            # Setup modulation schemes
            print("[DEBUG] Setting up modulation schemes...")
            self.modulation_schemes = {
                "QPSK": sn.mapping.Constellation("qam", num_bits_per_symbol=2),     # QPSK is 4-QAM (2 bits)
                "16QAM": sn.mapping.Constellation("qam", num_bits_per_symbol=4),    # 16QAM (4 bits)
                "64QAM": sn.mapping.Constellation("qam", num_bits_per_symbol=6)     # 64QAM (6 bits)
            }

            self.mappers = {
                mod: sn.mapping.Mapper(constellation=scheme)
                for mod, scheme in self.modulation_schemes.items()
            }

            # Initialize channel model
            print("[DEBUG] Setting up channel model...")
            self.channel_model = RayleighBlockFading(
                num_rx=1,
                num_rx_ant=self.system_params.num_rx_antennas,
                num_tx=1,
                num_tx_ant=self.system_params.num_tx_antennas,
                dtype=tf.complex64
            )

            print("[DEBUG] Sionna components setup completed successfully")

        except Exception as e:
            print(f"[ERROR] Failed to setup Sionna components: {str(e)}")
            raise

    def _generate_batch_data(self, batch_size: int, batch_idx: int = 0, modulation: str = "QPSK") -> Dict[str, tf.Tensor]:
        try:
            # Add this check at the start of the method
            if modulation not in self.modulation_schemes:
                raise ValueError(f"Unsupported modulation scheme: {modulation}")

            # Generate random bits for transmission
            bits_per_batch = batch_size * self.system_params.num_tx_antennas * self.system_params.num_bits_per_symbol
            bits = tf.random.uniform(
                shape=[bits_per_batch],
                minval=0,
                maxval=2,
                dtype=tf.int32
            )
            
            # Reshape bits for QPSK modulation
            bits = tf.reshape(bits, [batch_size, self.system_params.num_tx_antennas, self.system_params.num_bits_per_symbol])
            
            # Create QPSK constellation points
            constellation_points = tf.constant([
                1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j
            ], dtype=tf.complex64) / tf.cast(tf.sqrt(2.0), tf.complex64)
            
            # Map bits to QPSK symbols
            bits_concat = tf.reshape(bits, [-1, self.system_params.num_bits_per_symbol])
            symbol_indices = bits_concat[:, 0] * 2 + bits_concat[:, 1]
            tx_symbols = tf.gather(constellation_points, symbol_indices)
            tx_symbols = tf.reshape(tx_symbols, [batch_size, self.system_params.num_tx_antennas])
            
            # Generate channel response with correct shape
            h_real = tf.random.normal([batch_size, self.system_params.num_rx_antennas, self.system_params.num_tx_antennas], dtype=tf.float32)
            h_imag = tf.random.normal([batch_size, self.system_params.num_rx_antennas, self.system_params.num_tx_antennas], dtype=tf.float32)
            channel_response = tf.complex(h_real, h_imag) / tf.sqrt(2.0 * float(self.system_params.num_tx_antennas))
            
            # Reshape tx_symbols for matrix multiplication
            tx_symbols_expanded = tf.expand_dims(tx_symbols, axis=-1)  # Shape: [batch_size, num_tx_antennas, 1]
            
            # Apply channel - matrix multiplication
            y_without_noise = tf.matmul(channel_response, tx_symbols_expanded)  # Shape: [batch_size, num_rx_antennas, 1]
            y_without_noise = tf.squeeze(y_without_noise, axis=-1)  # Shape: [batch_size, num_rx_antennas]
            
            # Generate SNR values
            snr_db = tf.random.uniform(
                [batch_size],
                minval=self.system_params.min_snr_db,
                maxval=self.system_params.max_snr_db,
                dtype=tf.float32
            )
            
            # Convert SNR from dB to linear scale
            snr_linear = tf.pow(10.0, snr_db / 10.0)
            noise_power = 1.0 / snr_linear
            
            # Generate and add noise with proper broadcasting
            noise_power_expanded = tf.expand_dims(noise_power, axis=-1)  # Shape: [batch_size, 1]
            noise_std = tf.sqrt(noise_power_expanded / 2.0)
            
            # Generate complex noise with correct shape
            noise = tf.complex(
                tf.random.normal([batch_size, self.system_params.num_rx_antennas], dtype=tf.float32) * noise_std,
                tf.random.normal([batch_size, self.system_params.num_rx_antennas], dtype=tf.float32) * noise_std
            )
            
            # Add noise to received signals
            rx_symbols = y_without_noise + noise
            
            return {
                'channel_response': tf.cast(channel_response, tf.complex64),
                'tx_symbols': tf.cast(tx_symbols, tf.complex64),
                'rx_symbols': tf.cast(rx_symbols, tf.complex64),
                'snr_db': tf.cast(snr_db, tf.float32)
            }
            
        except Exception as e:
            self.logger.warning(f"Error generating batch {batch_idx}: {str(e)}")
            raise

    def _generate_batch_data(self, batch_size: int, batch_idx: int = 0, modulation: str = "QPSK") -> Dict[str, tf.Tensor]:
        try:
            # Get the appropriate mapper
            mapper = self.mappers[modulation]
            
            # Get bits per symbol from the mapper's constellation
            bits_per_symbol = mapper.constellation.num_bits_per_symbol
            
            # Calculate total number of bits needed
            num_bits = batch_size * self.system_params.num_tx_antennas * bits_per_symbol
            
            # Generate random bits
            bits = tf.random.uniform(
                [num_bits],
                minval=0,
                maxval=2,
                dtype=tf.int32
            )
            
            # Reshape bits to [total_symbols, bits_per_symbol] format
            total_symbols = num_bits // bits_per_symbol
            bits_reshaped = tf.reshape(bits, [total_symbols, bits_per_symbol])
            
            # Map bits to symbols using Sionna
            tx_symbols = mapper(bits_reshaped)
            tx_symbols = tf.cast(tx_symbols, tf.complex64)
            
            # Reshape symbols to [batch_size, num_tx_antennas]
            tx_symbols = tf.reshape(tx_symbols, 
                [batch_size, self.system_params.num_tx_antennas])

            # Generate SNR values with proper range
            snr_db = tf.random.uniform(
                [batch_size],
                minval=max(15.0, self.system_params.min_snr_db),
                maxval=self.system_params.max_snr_db,
                dtype=tf.float32
            )

            # Generate channel response using Sionna with required parameters
            h_real = tf.random.normal(
                [batch_size, self.system_params.num_rx_antennas, self.system_params.num_tx_antennas],
                dtype=tf.float32
            )
            h_imag = tf.random.normal(
                [batch_size, self.system_params.num_rx_antennas, self.system_params.num_tx_antennas],
                dtype=tf.float32
            )
            
            # Combine into complex tensor
            h = tf.complex(h_real, h_imag) / tf.cast(tf.sqrt(2.0), tf.complex64)
            
            # Log channel stats with proper casting
            self.mimo_logger.log_channel_stats(h, snr_db)

            # Apply channel
            noise_variance = tf.pow(10.0, -snr_db/10.0)
            # Reshape noise_variance to match required dimensions
            noise_variance = tf.reshape(noise_variance, [-1, 1])  # Shape: [batch_size, 1]
            
            # Generate noise with proper shape
            noise_real = tf.random.normal(
                [batch_size, self.system_params.num_rx_antennas],  # Match rx_symbols shape
                stddev=tf.sqrt(noise_variance/2)
            )
            noise_imag = tf.random.normal(
                [batch_size, self.system_params.num_rx_antennas],  # Match rx_symbols shape
                stddev=tf.sqrt(noise_variance/2)
            )
            noise = tf.complex(noise_real, noise_imag)

            # Apply channel with correct shape handling
            # Reshape tx_symbols for matrix multiplication
            tx_symbols_expanded = tf.expand_dims(tx_symbols, axis=-1)  # Shape: [batch_size, num_tx_antennas, 1]
            
            # Perform matrix multiplication and reshape
            rx_without_noise = tf.matmul(h, tx_symbols_expanded)  # Shape: [batch_size, num_rx_antennas, 1]
            rx_without_noise = tf.squeeze(rx_without_noise, axis=-1)  # Shape: [batch_size, num_rx_antennas]
            
            # Add noise (shapes should now match)
            rx_symbols = rx_without_noise + noise  # Both should be [batch_size, num_rx_antennas]

            return {
                'channel_response': h,  # Shape: [batch_size, num_rx_antennas, num_tx_antennas]
                'tx_symbols': tx_symbols,  # Shape: [batch_size, num_tx_antennas]
                'rx_symbols': rx_symbols,  # Shape: [batch_size, num_rx_antennas]
                'snr_db': tf.cast(snr_db, tf.float32)  # Shape: [batch_size]
            }

        except Exception as e:
            self.logger.error(f"Error in batch generation: {str(e)}")
            raise

    def _get_64qam_constellation(self):
        """Generate normalized 64QAM constellation points."""
        points = []
        for i in range(-7, 8, 2):
            for q in range(-7, 8, 2):
                points.append(complex(i, q))
        return tf.constant(points, dtype=tf.complex64) / tf.cast(tf.sqrt(42.0), tf.complex64)

    def generate_dataset(self, save_path: str = 'dataset/mimo_dataset.h5') -> str:
        """
        Generate and save the MIMO dataset to an HDF5 file.
        
        Args:
            save_path (str): Path where the dataset will be saved
            
        Returns:
            str: Path to the saved dataset file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Calculate number of batches
            num_batches = self.system_params.total_samples // self.system_params.batch_size
            
            with h5py.File(save_path, 'w') as f:
                # Create channel data group
                channel_group = f.create_group('channel_data')
                
                # Create datasets with full size
                channel_response_dataset = channel_group.create_dataset(
                    'channel_response',
                    shape=(self.system_params.total_samples, 
                        self.system_params.num_rx_antennas,
                        self.system_params.num_tx_antennas),
                    dtype=np.complex64
                )
                
                tx_symbols_dataset = channel_group.create_dataset(
                    'tx_symbols',
                    shape=(self.system_params.total_samples, 
                        self.system_params.num_tx_antennas),
                    dtype=np.complex64
                )
                
                rx_symbols_dataset = channel_group.create_dataset(
                    'rx_symbols',
                    shape=(self.system_params.total_samples, 
                        self.system_params.num_rx_antennas),
                    dtype=np.complex64
                )
                
                snr_db_dataset = channel_group.create_dataset(
                    'snr_db',
                    shape=(self.system_params.total_samples,),
                    dtype=np.float32
                )
                
                # Generate and save data in batches
                from tqdm import tqdm
                for batch_idx in tqdm(range(num_batches), desc="Generating samples"):
                    start_idx = batch_idx * self.system_params.batch_size
                    end_idx = start_idx + self.system_params.batch_size
                    
                    try:
                        batch_data = self._generate_batch_data(
                            batch_size=self.system_params.batch_size,
                            batch_idx=batch_idx
                        )
                        
                        # Save batch data to datasets
                        channel_response_dataset[start_idx:end_idx] = batch_data['channel_response'].numpy()
                        tx_symbols_dataset[start_idx:end_idx] = batch_data['tx_symbols'].numpy()
                        rx_symbols_dataset[start_idx:end_idx] = batch_data['rx_symbols'].numpy()
                        snr_db_dataset[start_idx:end_idx] = batch_data['snr_db'].numpy()
                        
                        # Add progress logging every 10 batches
                        if batch_idx % 10 == 0:
                            self.mimo_logger.log_generation_progress(
                                batch_idx=batch_idx,
                                total_batches=num_batches,
                                current_metrics=batch_data
                            )
                        
                    except Exception as e:
                        self.logger.error(f"Error generating batch {batch_idx}: {str(e)}")
                        raise
                
                # Save system parameters
                config_group = f.create_group('config')
                for key, value in self.system_params.get_config_dict().items():
                    if isinstance(value, (int, float, str)):
                        config_group.attrs[key] = value
                
                self.logger.info(f"Dataset successfully generated and saved to {save_path}")
                return save_path
                
        except Exception as e:
            self.logger.error(f"Error generating dataset: {str(e)}")
            raise

    def verify_complex_data(self, file_path):
        """Verify complex data types in the dataset.
        
        Args:
            file_path (str): Path to the HDF5 dataset file
        """
        with h5py.File(file_path, 'r') as f:
            channel_response = f['channel_data']['channel_response'][:]
            print("Channel Response dtype:", channel_response.dtype)
            print("Contains complex values:", np.iscomplexobj(channel_response))
            print("Shape:", channel_response.shape)
                    
    def _validate_batch_data(self, channel_response: tf.Tensor, metrics: Dict[str, tf.Tensor]) -> bool:
        """
        Validate generated batch data
        
        Args:
            channel_response: Generated channel responses
            metrics: Dictionary of calculated metrics
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        try:
            # Check for NaN or Inf values
            if tf.reduce_any(tf.math.is_nan(channel_response)) or \
            tf.reduce_any(tf.math.is_inf(channel_response)):
                self.logger.warning("Invalid values in channel response")
                return False
                
            # Validate metrics
            for metric_name, metric_data in metrics.items():
                if tf.reduce_any(tf.math.is_nan(metric_data)) or \
                tf.reduce_any(tf.math.is_inf(metric_data)):
                    self.logger.warning(f"Invalid values in metric: {metric_name}")
                    return False
                # Validate channel response shape
            if not validate_mimo_tensor_shapes(
                channel_response,
                self.system_params.num_tx_antennas,
                self.system_params.num_rx_antennas,
                self.system_params.batch_size
            ):
                return False
        
            # Validate metrics
            if not validate_mimo_metrics(
                metrics,
                self.system_params.batch_size,
                self.system_params.num_tx_antennas,
                self.system_params.num_rx_antennas
                ):
                return False
            
            self.mimo_logger.log_channel_stats(channel_response)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
        return False
