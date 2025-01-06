# class ChannelModelManager
# core/channel_model.py
# MIMO Channel Model Generation and Management
# Focuses on generating realistic MIMO channel responses for beamforming optimization tasks.
#
# Key Features:
# - Integrates with the Sionna library for accurate and efficient channel modeling.
# - Supports Rayleigh fading and static user scenarios based on the simulation plan.
# - Configures antenna arrays and resource grid parameters for flexible MIMO setups.
# - Provides channel realization generation aligned with system parameters (e.g., 4x4 MIMO, 64 subcarriers).
#
# Simplifications:
# - Removed support for mobility-based models (e.g., Doppler effects) to focus on static user scenarios.
# - Streamlined antenna and grid configurations for reproducibility and ease of use.
#
# Updates:
# - Aligned channel modeling with the simulation plan to ensure compatibility and efficiency.
# - Simplified resource grid setup for static, single-slot OFDM configurations.
# - Enhanced logging and validation for channel response consistency.
#
# Scope:
# - This module focuses on channel generation for training reinforcement learning models in MIMO systems.
# - Advanced mobility models and real-time updates are excluded for simplicity.

import tensorflow as tf
import numpy as np
from sionna.channel import RayleighBlockFading
from sionna.channel.tr38901 import AntennaArray
from sionna.ofdm import ResourceGrid
from sionna.mapping import Mapper
from typing import Tuple, Optional, Dict
from config.system_parameters import SystemParameters
from core.path_loss_model import PathLossManager
from utill.tensor_shape_validator import normalize_complex_tensor
from utill.logging_config import LoggerManager

class ChannelModelManager:
    """Simplified channel model management for MIMO systems"""
    
    def __init__(self, system_params: Optional[SystemParameters] = None):
        self.system_params = system_params or SystemParameters()
        self.logger = LoggerManager.get_logger(__name__)
        
        # Initialize components
        self._setup_antenna_arrays()
        self._setup_resource_grid()
        self._setup_channel_model()
        self.path_loss_manager = PathLossManager(self.system_params)
        
        # Validation thresholds
        self.validation_thresholds = {
            'path_loss': {'min': 20.0, 'max': 160.0},
            'channel_response': {'min': -100.0, 'max': 100.0},
            'distance': {'min': 1.0, 'max': 500.0}  # Updated max distance
        }

    def _setup_antenna_arrays(self):
        """Configure antenna arrays for transmitter and receiver."""
        try:
            self.tx_array = AntennaArray(
                num_rows=1,
                num_cols=self.system_params.num_tx_antennas,  # Updated parameter name
                polarization="single",
                carrier_frequency=self.system_params.carrier_frequency,
                horizontal_spacing=self.system_params.element_spacing  # Added spacing
            )

            self.rx_array = AntennaArray(
                num_rows=1,
                num_cols=self.system_params.num_rx_antennas,  # Updated parameter name
                polarization="single",
                carrier_frequency=self.system_params.carrier_frequency,
                horizontal_spacing=self.system_params.element_spacing  # Added spacing
            )
            
            self.logger.info("Antenna arrays configured successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup antenna arrays: {str(e)}")
            raise

    def _setup_resource_grid(self):
        """Configure OFDM resource grid."""
        try:
            self.resource_grid = ResourceGrid(
                num_ofdm_symbols=self.system_params.num_ofdm_symbols,
                fft_size=self.system_params.num_subcarriers,
                subcarrier_spacing=self.system_params.subcarrier_spacing,
                num_tx=self.system_params.num_tx_antennas,  # Updated parameter name
                num_streams_per_tx=self.system_params.num_streams,  # Added streams
                cyclic_prefix_length=self.system_params.num_subcarriers // 4,  # Dynamic CP length
            )
            
            self.logger.info("Resource grid configured successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup resource grid: {str(e)}")
            raise

    def _setup_channel_model(self):
        """Initialize Rayleigh block fading channel model."""
        try:
            self.channel_model = RayleighBlockFading(
                num_rx=self.system_params.num_rx_antennas,  # Updated parameter name
                num_tx=self.system_params.num_tx_antennas,  # Updated parameter name
                num_time_steps=self.system_params.num_ofdm_symbols,
                dtype=tf.complex64
            )
            
            self.logger.info("Channel model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup channel model: {str(e)}")
            raise

    def generate_channel_samples(self, batch_size: int, snr_db: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Generate channel samples with path loss and noise."""
        try:
            # Generate and validate distances
            distances = tf.random.uniform(
                [batch_size], 
                self.validation_thresholds['distance']['min'],
                self.validation_thresholds['distance']['max']
            )
            
            # Calculate path loss
            path_loss = self.path_loss_manager.calculate_free_space_path_loss(distances)
            path_loss_linear = tf.pow(10.0, -path_loss / 10.0)
            
            # Generate and normalize channel response
            h = self.channel_model()
            h_normalized = normalize_complex_tensor(h)
            
            # Apply path loss
            h_with_path_loss = h_normalized * tf.sqrt(tf.reshape(path_loss_linear, [-1, 1, 1]))
            
            # Add noise based on SNR
            noise_power = tf.pow(10.0, -snr_db / 10.0)
            noise = tf.complex(
                tf.random.normal(tf.shape(h_with_path_loss), stddev=tf.sqrt(noise_power / 2.0)),
                tf.random.normal(tf.shape(h_with_path_loss), stddev=tf.sqrt(noise_power / 2.0))
            )
            
            noisy_channel = h_with_path_loss + noise
            
            # Validate outputs
            self._validate_channel_response(h_with_path_loss)
            
            return {
                "perfect_channel": h_with_path_loss,
                "noisy_channel": noisy_channel,
                "path_loss": path_loss,
                "distances": distances
            }
            
        except Exception as e:
            self.logger.error(f"Error generating channel samples: {str(e)}")
            raise

    def _validate_channel_response(self, channel_response: tf.Tensor):
        """Validate channel response values."""
        max_value = tf.reduce_max(tf.abs(channel_response))
        min_value = tf.reduce_min(tf.abs(channel_response))
        
        if max_value > self.validation_thresholds['channel_response']['max'] or \
        min_value < self.validation_thresholds['channel_response']['min']:
            self.logger.warning(f"Channel response values outside expected range: "
                            f"min={min_value}, max={max_value}")