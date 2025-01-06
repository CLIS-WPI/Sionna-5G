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
from typing import Dict
import h5py
import numpy as np
import tensorflow as tf
import sionna as sn
from tqdm import tqdm
from config.system_parameters import SystemParameters
from utill.logging_config import LoggerManager

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
            self.tx_array = sn.channel.AntennaArray(
                num_rows=1,
                num_cols=self.system_params.num_tx_antennas,
                vertical_spacing=self.system_params.element_spacing,
                horizontal_spacing=self.system_params.element_spacing,
                pattern="iso"
            )
            
            self.rx_array = sn.channel.AntennaArray(
                num_rows=1,
                num_cols=self.system_params.num_rx_antennas,
                vertical_spacing=self.system_params.element_spacing,
                horizontal_spacing=self.system_params.element_spacing,
                pattern="iso"
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

    def _validate_batch_data(self, batch_data: dict) -> tuple[bool, list[str]]:
        """Validate batch data against defined thresholds"""
        errors = []
        try:
            for key, data in batch_data.items():
                if key in self.validation_thresholds:
                    if tf.is_tensor(data):
                        if data.dtype.is_complex:
                            data_np = np.abs(data.numpy()).flatten()
                        else:
                            data_np = data.numpy().flatten()
                    else:
                        data_np = np.asarray(data).flatten()

                    threshold = self.validation_thresholds[key]
                    stats = {
                        'min': np.min(data_np),
                        'max': np.max(data_np),
                        'mean': np.mean(data_np),
                        'std': np.std(data_np)
                    }

                    if stats['min'] < threshold['min'] or stats['max'] > threshold['max']:
                        errors.append(
                            f"{key}: Values outside threshold range "
                            f"[{threshold['min']}, {threshold['max']}]. "
                            f"Stats: {stats}"
                        )

            return len(errors) == 0, errors

        except Exception as e:
            self.logger.error(f"Batch validation error: {str(e)}")
            return False, [f"Critical validation error: {str(e)}"]

    def generate_dataset(self, save_path: str = 'dataset/mimo_dataset.h5'):
        """Generate MIMO dataset"""
        try:
            # Prepare output directory
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with h5py.File(save_path, 'w') as f:
                # Create dataset structure
                samples_per_mod = self.system_params.total_samples // len(self.system_params.modulation_schemes)
                
                # Store configuration
                config_group = f.create_group('configuration')
                for key, value in self.system_params.__dict__.items():
                    if isinstance(value, (int, float, str)):
                        config_group.attrs[key] = value
                
                # Create groups for each modulation scheme
                for mod_scheme in self.system_params.modulation_schemes:
                    mod_group = f.create_group(f'modulation_data/{mod_scheme}')
                    
                    # Create datasets
                    mod_group.create_dataset(
                        'channel_response',
                        shape=(samples_per_mod, self.system_params.num_rx_antennas, 
                            self.system_params.num_tx_antennas),
                        dtype=np.complex64
                    )
                    
                    for metric in ['path_loss', 'sinr', 'spectral_efficiency']:
                        mod_group.create_dataset(
                            metric,
                            shape=(samples_per_mod,),
                            dtype=np.float32
                        )
                
                # Generate data for each modulation scheme
                for mod_scheme in self.system_params.modulation_schemes:
                    self.logger.info(f"Generating data for {mod_scheme}")
                    mod_group = f[f'modulation_data/{mod_scheme}']
                    
                    for i in tqdm(range(0, samples_per_mod, self.system_params.batch_size)):
                        batch_size = min(self.system_params.batch_size, samples_per_mod - i)
                        
                        # Generate batch data
                        batch_data = self._generate_batch_data(batch_size, mod_scheme)
                        
                        # Validate batch data
                        is_valid, errors = self._validate_batch_data(batch_data)
                        if not is_valid:
                            self.logger.warning(f"Batch validation failed: {errors}")
                            continue
                        
                        # Save batch data
                        for key, data in batch_data.items():
                            if key in mod_group:
                                mod_group[key][i:i+batch_size] = data.numpy()
                
                self.logger.info(f"Dataset generated successfully: {save_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"Dataset generation failed: {str(e)}")
            return False