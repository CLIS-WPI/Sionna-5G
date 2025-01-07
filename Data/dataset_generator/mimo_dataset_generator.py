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
from core.path_loss_model import PathLossManager


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

    def _generate_batch_data(self, batch_size: int, mod_scheme: str = 'QPSK') -> Dict[str, tf.Tensor]:
        """
        Generate a batch of MIMO channel data
        
        Args:
            batch_size (int): Size of the batch to generate
            mod_scheme (str, optional): Modulation scheme to use. Defaults to 'QPSK'
            
        Returns:
            Dict[str, tf.Tensor]: Dictionary containing generated data
        """
        try:
            # Generate distances first
            distances = tf.random.uniform([batch_size], minval=10.0, maxval=500.0)
            
            # Calculate path loss using PathLossManager
            path_loss_db = self.path_loss_manager.calculate_free_space_path_loss(distances)
            
            # Generate complex channel response
            channel_response = tf.complex(
                tf.random.normal([batch_size, self.system_params.num_rx_antennas, 
                                self.system_params.num_tx_antennas], dtype=tf.float32),
                tf.random.normal([batch_size, self.system_params.num_rx_antennas, 
                                self.system_params.num_tx_antennas], dtype=tf.float32)
            )
            
            # Ensure complex64 type
            channel_response = tf.cast(channel_response, tf.complex64)
            
            # Convert path loss to linear scale and reshape for broadcasting
            path_loss_linear = tf.pow(10.0, -path_loss_db/20.0)
            path_loss_shaped = tf.cast(
                tf.reshape(path_loss_linear, [-1, 1, 1]),
                tf.complex64
            )
            
            # Apply path loss to channel response
            channel_response = channel_response * path_loss_shaped
            
            # Calculate metrics using the MetricsCalculator
            metrics = self.metrics_calculator.calculate_mimo_metrics(
                channel_response=channel_response,
                snr_db=tf.random.uniform([batch_size], 
                                    self.system_params.min_snr_db,
                                    self.system_params.max_snr_db)
            )
            
            return {
                'channel_response': channel_response,  # complex64
                'path_loss_db': tf.cast(path_loss_db, tf.float32),  # Now properly defined
                'distances': tf.cast(distances, tf.float32),
                'modulation_scheme': mod_scheme,
                'effective_snr': metrics['effective_snr'],
                **metrics  # Include calculated metrics
            }
            
        except Exception as e:
            self.logger.error(f"Batch generation failed: {str(e)}")
            raise

    def generate_dataset(self, save_path: str = 'dataset/mimo_dataset.h5'):
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with h5py.File(save_path, 'w') as f:
                # Create main data group
                data_group = f.create_group('channel_data')
                
                # Define datasets to create
                required_datasets = {
                    'channel_response': (self.system_params.total_samples, 
                                    self.system_params.num_rx_antennas,
                                    self.system_params.num_tx_antennas),
                    'path_loss_db': (self.system_params.total_samples,),
                    'distances': (self.system_params.total_samples,),
                    'spectral_efficiency': (self.system_params.total_samples,),
                    'effective_snr': (self.system_params.total_samples,),  # Add effective_snr
                    'eigenvalues': (self.system_params.total_samples,),
                    'condition_number': (self.system_params.total_samples,)
                }
                
                # Create datasets
                for name, shape in required_datasets.items():
                    dtype = np.complex64 if name == 'channel_response' else np.float32
                    data_group.create_dataset(name, shape=shape, dtype=dtype)
                
                # Generate data in batches
                for i in tqdm(range(0, self.system_params.total_samples, 
                                self.system_params.batch_size)):
                    batch_size = min(self.system_params.batch_size, 
                                self.system_params.total_samples - i)
                    
                    try:
                        batch_data = self._generate_batch_data(batch_size=batch_size)
                        
                        # Store all batch data
                        for key, value in batch_data.items():
                            if key in data_group:
                                data_group[key][i:i+batch_size] = value.numpy()
                                
                    except Exception as batch_error:
                        self.logger.warning(f"Error generating batch {i}: {str(batch_error)}")
                        continue
                
                self.logger.info(f"Dataset generated successfully: {save_path}")
                
                # Verify dataset integrity
                checker = MIMODatasetIntegrityChecker(save_path, self.system_params)
                integrity_report = checker.check_dataset_integrity()
                
                if not integrity_report['overall_status']:
                    self.logger.error("Dataset integrity check failed")
                    if 'errors' in integrity_report:
                        for error in integrity_report['errors']:
                            self.logger.error(error)
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
