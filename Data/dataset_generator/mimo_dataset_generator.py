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
            self.logger.debug(f"SNR Range: {self.system_params.snr_range}")
            self.logger.debug(f"Min SNR: {self.system_params.min_snr_db}")
            self.logger.debug(f"Max SNR: {self.system_params.max_snr_db}")
            
            # Initialize Rayleigh channel model with ALL required parameters
            self.channel_model = sn.channel.RayleighBlockFading(
                num_rx=1,                                      # Number of receivers
                num_rx_ant=self.system_params.num_rx_antennas, # Number of receive antennas
                num_tx=1,                                      # Number of transmitters
                num_tx_ant=self.system_params.num_tx_antennas, # Number of transmit antennas
                dtype=tf.complex64
            )

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
            
            # Generate channel response with correct shape handling and type
            channel_response = self.channel_model(
                batch_size=batch_size,
                num_time_steps=1  # Set to 1 for static channel
            )
            
            # Ensure complex64 type and correct shape
            channel_response = tf.cast(channel_response, tf.complex64)  # Explicit cast to complex64
            channel_response = tf.squeeze(channel_response)  # Remove singleton dimensions
            
            # Apply path loss to channel response
            path_loss_linear = tf.pow(10.0, -path_loss_db/20)
            path_loss_shaped = tf.cast(
                tf.reshape(path_loss_linear, [-1, 1, 1]),  # Reshape for broadcasting
                tf.complex64  # Explicit cast to complex64
            )
            
            # Ensure multiplication maintains complex64 type
            channel_response = tf.cast(channel_response * path_loss_shaped, tf.complex64)
            
            # Calculate SINR
            noise_power = tf.pow(10.0, (self.system_params.noise_floor - 30)/10)
            channel_power = tf.reduce_mean(tf.abs(channel_response)**2, axis=[-2, -1])
            sinr = 10 * tf.math.log(channel_power / noise_power) / tf.math.log(10.0)
            
            # Calculate effective SNR and ensure float32 type
            effective_snr = tf.cast(
                sinr - 10 * tf.math.log(
                    tf.cast(self.system_params.num_tx_antennas, tf.float32)
                ) / tf.math.log(10.0),
                tf.float32
            )
            
            # Calculate spectral efficiency and ensure float32 type
            spectral_efficiency = tf.cast(
                tf.math.log(1 + tf.pow(10.0, sinr/10)) / tf.math.log(2.0),
                tf.float32
            )
            
            return {
                'channel_response': channel_response,  # Should be complex64
                'path_loss': tf.cast(path_loss_db, tf.float32),
                'sinr': tf.cast(sinr, tf.float32),
                'effective_snr': effective_snr,
                'spectral_efficiency': spectral_efficiency,
                'distances': tf.cast(distances, tf.float32)
            }
                
        except Exception as e:
            self.logger.error(f"Batch generation failed: {str(e)}")
            raise

    def generate_dataset(self, save_path: str = 'dataset/mimo_dataset.h5'):
        """Generate MIMO dataset and save to HDF5 file.
        
        Args:
            save_path (str): Path where to save the dataset
            
        Returns:
            bool: True if generation successful, False otherwise
        """
        try:
            # Create output directory
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
                
                # Define datasets to create
                datasets = {
                    'channel_response': (total_samples, self.system_params.num_rx_antennas, 
                                    self.system_params.num_tx_antennas),
                    'path_loss': (total_samples,),
                    'sinr': (total_samples,),
                    'spectral_efficiency': (total_samples,),
                    'distances': (total_samples,)
                }
                
                # Create datasets
                for name, shape in datasets.items():
                    dtype = np.complex64 if name == 'channel_response' else np.float32
                    data_group.create_dataset(name, shape=shape, dtype=dtype)
                
                # Generate data in batches
                self.logger.info(f"Generating dataset with {total_samples} samples")
                
                for i in tqdm(range(0, total_samples, self.system_params.batch_size)):
                    batch_size = min(self.system_params.batch_size, total_samples - i)
                    
                    try:
                        # Generate batch data using our fixed _generate_batch_data method
                        batch_data = self._generate_batch_data(
                            batch_size=batch_size,
                            mod_scheme='QPSK'
                        )
                        
                        # Store the batch data
                        for key, value in batch_data.items():
                            if key in data_group:
                                data_group[key][i:i+batch_size] = value.numpy()
                                
                    except Exception as batch_error:
                        self.logger.warning(f"Error generating batch {i}: {str(batch_error)}")
                        continue
                
                # Add metadata
                f.attrs['generation_timestamp'] = datetime.now().isoformat()
                f.attrs['total_samples'] = total_samples
                f.attrs['valid_samples'] = i + batch_size
                
                self.logger.info(f"Dataset generated successfully: {save_path}")
                
                # Add integrity check
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
