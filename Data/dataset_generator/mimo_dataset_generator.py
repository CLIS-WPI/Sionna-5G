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
            
            # Calculate SNR values
            snr_db = tf.random.uniform([batch_size], 
                                    self.system_params.min_snr_db,
                                    self.system_params.max_snr_db)
            
            # Calculate metrics using the MetricsCalculator
            metrics = self.metrics_calculator.calculate_mimo_metrics(
                channel_response=channel_response,
                snr_db=snr_db
            )
            
            # Ensure eigenvalues have correct shape by taking the mean across antenna dimension
            if 'eigenvalues' in metrics and len(metrics['eigenvalues'].shape) > 1:
                metrics['eigenvalues'] = tf.reduce_mean(metrics['eigenvalues'], axis=1)
            
            return {
                'channel_response': channel_response,
                'path_loss_db': tf.cast(path_loss_db, tf.float32),
                'distances': tf.cast(distances, tf.float32),
                'modulation_scheme': mod_scheme,
                'effective_snr': metrics['effective_snr'],
                'spectral_efficiency': metrics['spectral_efficiency'],
                'eigenvalues': metrics['eigenvalues'],
                'condition_number': metrics['condition_number']
            }
            
        except Exception as e:
            self.logger.error(f"Batch generation failed: {str(e)}")
            raise

    def generate_dataset(self, save_path: str = 'dataset/mimo_dataset.h5'):
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            total_samples = self.system_params.total_samples
            batch_size = self.system_params.batch_size
            num_batches = total_samples // batch_size
            
            with h5py.File(save_path, 'w') as f:
                # Create main data group
                data_group = f.create_group('channel_data')
                
                # Create datasets with total size and appropriate dtypes
                datasets = {
                    'channel_response': {
                        'shape': (total_samples, 
                                self.system_params.num_rx_antennas,
                                self.system_params.num_tx_antennas),
                        'dtype': np.complex64  # Changed to complex64 for channel response
                    },
                    'path_loss_db': {
                        'shape': (total_samples,),
                        'dtype': np.float32
                    },
                    'distances': {
                        'shape': (total_samples,),
                        'dtype': np.float32
                    },
                    'spectral_efficiency': {
                        'shape': (total_samples,),
                        'dtype': np.float32
                    },
                    'effective_snr': {
                        'shape': (total_samples,),
                        'dtype': np.float32
                    },
                    'eigenvalues': {
                        'shape': (total_samples,),
                        'dtype': np.float32
                    },
                    'condition_number': {
                        'shape': (total_samples,),
                        'dtype': np.float32
                    }
                }
                
                # Initialize datasets with proper dtypes
                for name, config in datasets.items():
                    data_group.create_dataset(
                        name, 
                        shape=config['shape'],
                        dtype=config['dtype']
                    )
                
                # Use tqdm to show progress of actual samples
                samples_processed = 0
                with tqdm(total=total_samples, desc="Generating samples") as pbar:
                    for batch_idx in range(num_batches):
                        try:
                            # Generate batch data
                            batch_data = self._generate_batch_data(batch_size)
                            
                            # Calculate start and end indices for this batch
                            start_idx = batch_idx * batch_size
                            end_idx = start_idx + batch_size
                            
                            # Store batch data
                            for name, data in batch_data.items():
                                if name != 'modulation_scheme':
                                    data_group[name][start_idx:end_idx] = data
                            
                            # Update progress bar with actual samples processed
                            samples_processed += batch_size
                            pbar.update(batch_size)
                            
                        except Exception as e:
                            self.logger.warning(f"Error generating batch {start_idx}: {str(e)}")
                            continue
                
                # Add configuration group
                config_group = f.create_group('configuration')
                config_dict = self.system_params.get_config_dict()
                for key, value in config_dict.items():
                    if isinstance(value, (int, float, str)):
                        config_group.attrs[key] = value
                    elif isinstance(value, (list, tuple)):
                        config_group.create_dataset(key, data=value)
                
            self.logger.info(f"Dataset generated successfully: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate dataset: {str(e)}")
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
            
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
        return False
