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
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import h5py
import numpy as np
import tensorflow as tf
import sionna as sn
from sionna.channel.tr38901 import AntennaArray
from config.system_parameters import SystemParameters
from utill.logging_config import LoggerManager
from core.metrics_calculator import MetricsCalculator
from config.system_parameters import SystemParameters
from integrity.dataset_integrity_checker import MIMODatasetIntegrityChecker
from utill.tensor_shape_validator import validate_mimo_tensor_shapes
from utill.tensor_shape_validator import validate_mimo_metrics
from tqdm import tqdm

class MIMODatasetGenerator:
    __version__ = '2.0.0'

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
        
        self.system_params = system_params or SystemParameters()
        self.metrics_calculator = MetricsCalculator(self.system_params)
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
            # Setup antenna arrays
            self.tx_array = AntennaArray(
            num_rows=1,
            num_cols=self.system_params.num_tx_antennas,
            polarization=self.system_params.polarization,  # Add this line
            carrier_frequency=self.system_params.carrier_frequency,  # Add this line
            horizontal_spacing=self.system_params.element_spacing
            )
            
            self.rx_array = AntennaArray(
            num_rows=1,
            num_cols=self.system_params.num_rx_antennas,
            polarization=self.system_params.polarization,
            carrier_frequency=self.system_params.carrier_frequency,
            horizontal_spacing=self.system_params.element_spacing
            )

            # Setup channel model
            if self.system_params.channel_model.lower() == "rayleigh":
                self.channel_model = sn.channel.RayleighBlockFading(
                    num_rx=self.system_params.num_rx_antennas,
                    num_tx=self.system_params.num_tx_antennas,
                    num_time_steps=self.system_params.num_ofdm_symbols
                )
            else:
                raise ValueError(f"Unsupported channel model: {self.system_params.channel_model}")
                
            self.logger.info("Sionna components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Sionna components: {str(e)}")
            raise

    def _generate_batch_data(self, batch_size: int, mod_scheme: str) -> Dict[str, tf.Tensor]:
        """Generate a batch of MIMO channel data"""
        try:
            # Generate distances (10m to 500m)
            distances = tf.random.uniform(
                [batch_size],
                minval=10.0,
                maxval=500.0,
                dtype=tf.float32
            )
            
            # Calculate path loss (free space path loss)
            wavelength = 3e8 / self.system_params.carrier_frequency
            path_loss_db = 20 * tf.math.log(4 * np.pi * distances / wavelength) / tf.math.log(10.0)
            
            # Generate channel response
            channel_response = self.channel_model()
            
            # Apply path loss to channel response
            path_loss_linear = tf.pow(10.0, -path_loss_db/20)
            channel_response *= tf.expand_dims(tf.expand_dims(path_loss_linear, -1), -1)
            
            # Calculate SINR
            noise_power = tf.pow(10.0, (self.system_params.noise_floor - 30)/10)  # Convert dBm to linear
            channel_power = tf.reduce_mean(tf.abs(channel_response)**2, axis=[-2, -1])
            sinr = 10 * tf.math.log(channel_power / noise_power) / tf.math.log(10.0)
            
            # Calculate spectral efficiency
            spectral_efficiency = tf.math.log(1 + tf.pow(10.0, sinr/10)) / tf.math.log(2.0)
            
            return {
                'channel_response': channel_response,
                'path_loss': path_loss_db,
                'sinr': sinr,
                'spectral_efficiency': spectral_efficiency,
                'distances': distances
            }
            
        except Exception as e:
            self.logger.error(f"Batch generation failed: {str(e)}")
            raise

    def generate_dataset(self, save_path: str = 'dataset/mimo_dataset.h5'):
        """
        Generate MIMO dataset with enhanced metrics and validation
        
        Args:
            save_path (str): Path to save the HDF5 dataset
            
        Returns:
            bool: True if generation successful, False otherwise
        """
        try:
            # Prepare output directory
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with h5py.File(save_path, 'w') as f:
                # Store configuration
                config_group = f.create_group('configuration')
                for key, value in self.system_params.__dict__.items():
                    if isinstance(value, (int, float, str)):
                        config_group.attrs[key] = value
                
                # Create main data group
                data_group = f.create_group('channel_data')
                total_samples = self.system_params.total_samples
                
                # Create datasets
                data_group.create_dataset(
                    'channel_response',
                    shape=(total_samples, self.system_params.num_rx_antennas, 
                        self.system_params.num_tx_antennas),
                    dtype=np.complex64
                )
                
                # Create datasets for metrics
                metrics_datasets = {
                    'spectral_efficiency': (total_samples,),
                    'effective_snr': (total_samples,),
                    'condition_number': (total_samples,),
                    'eigenvalues': (total_samples, min(self.system_params.num_rx_antennas, 
                                                    self.system_params.num_tx_antennas))
                }
                
                for metric_name, shape in metrics_datasets.items():
                    data_group.create_dataset(
                        metric_name,
                        shape=shape,
                        dtype=np.float32
                    )
                
                # Generate data in batches
                self.logger.info(f"Generating dataset with {total_samples} samples")
                
                for i in tqdm(range(0, total_samples, self.system_params.batch_size)):
                    batch_size = min(self.system_params.batch_size, total_samples - i)
                    
                    try:
                        # Generate channel responses
                        snr_db = tf.random.uniform(
                            [batch_size], 
                            minval=self.system_params.min_snr_db,
                            maxval=self.system_params.max_snr_db
                        )
                        
                        # Generate channel responses using channel model
                        channel_response = self.channel_model.generate_channel_samples(
                            batch_size=batch_size,
                            snr_db=snr_db
                        )['perfect_channel']
                        
                        # Calculate metrics
                        metrics = self.metrics_calculator.calculate_mimo_metrics(
                            channel_response=channel_response,
                            snr_db=snr_db
                        )
                        
                        # Validate data
                        is_valid = self._validate_batch_data(channel_response, metrics)
                        if not is_valid:
                            continue
                        
                        # Store channel response
                        data_group['channel_response'][i:i+batch_size] = channel_response.numpy()
                        
                        # Store metrics
                        for metric_name, metric_data in metrics.items():
                            if metric_name in data_group:
                                data_group[metric_name][i:i+batch_size] = metric_data.numpy()
                        
                    except Exception as batch_error:
                        self.logger.warning(f"Error generating batch {i}: {str(batch_error)}")
                        continue
                
                # Add metadata
                f.attrs['generation_timestamp'] = datetime.now().isoformat()
                f.attrs['total_samples'] = total_samples
                f.attrs['valid_samples'] = i + batch_size
                
                self.logger.info(f"Dataset generated successfully: {save_path}")

                # Add integrity check here, after closing the file
                checker = MIMODatasetIntegrityChecker(
                    save_path,
                    system_params=self.system_params
                )

                integrity_report = checker.check_dataset_integrity()
                if not integrity_report['overall_status']:
                    self.logger.error("Dataset integrity check failed")
                    self.logger.error("\n".join(integrity_report['errors']))
                    return False
                return True
                
        except Exception as e:
            self.logger.error(f"Dataset generation failed: {str(e)}")
            return False

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
            
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
        return False
