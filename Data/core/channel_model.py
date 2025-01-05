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
    """
    Simplified channel model management for MIMO systems
    """

    def __init__(self, system_params: Optional[SystemParameters] = None):
        """
        Initialize the channel model manager.

        Args:
            system_params (Optional[SystemParameters]): System configuration.
        """
        self.system_params = system_params or SystemParameters()
        self.logger = LoggerManager.get_logger(__name__)

        # Initialize components
        self._setup_antenna_arrays()
        self._setup_resource_grid()
        self._setup_channel_model()
        self.path_loss_manager = PathLossManager(self.system_params)

    def _setup_antenna_arrays(self):
        """
        Configure antenna arrays for transmitter and receiver.
        """
        self.tx_array = AntennaArray(
            num_rows=self.system_params.num_tx,
            num_cols=1,
            polarization="single",
            carrier_frequency=self.system_params.carrier_frequency,
        )

        self.rx_array = AntennaArray(
            num_rows=self.system_params.num_rx,
            num_cols=1,
            polarization="single",
            carrier_frequency=self.system_params.carrier_frequency,
        )

    def _setup_resource_grid(self):
        """
        Configure OFDM resource grid.
        """
        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=self.system_params.num_ofdm_symbols,
            fft_size=self.system_params.num_subcarriers,
            subcarrier_spacing=self.system_params.subcarrier_spacing,
            num_tx=self.system_params.num_tx,
            num_streams_per_tx=1,
            cyclic_prefix_length=16,
        )

    def _setup_channel_model(self):
        """
        Initialize Rayleigh block fading channel model.
        """
        self.channel_model = RayleighBlockFading(
            num_rx=self.system_params.num_rx,
            num_tx=self.system_params.num_tx,
            dtype=tf.complex64,
        )

    def generate_channel_samples(self, batch_size: int, snr_db: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Generate channel samples with path loss and noise.

        Args:
            batch_size (int): Number of samples to generate.
            snr_db (tf.Tensor): SNR values in dB.

        Returns:
            Dict[str, tf.Tensor]: Dictionary containing perfect and noisy channel samples.
        """
        try:
            # Generate random distances in meters
            min_distance = 1.0
            max_distance = 100.0
            distances = tf.random.uniform([batch_size], min_distance, max_distance)

            # Validate and clip distances
            distances = tf.clip_by_value(distances, min_distance, max_distance)

            # Log distance stats for debugging
            self.logger.debug(f"Generated distances (min: {tf.reduce_min(distances).numpy()}, "
                            f"max: {tf.reduce_max(distances).numpy()})")

            # Calculate path loss using FSPL
            path_loss = self.path_loss_manager.calculate_free_space_path_loss(distances)
            path_loss_linear = tf.pow(10.0, -path_loss / 10.0)

            # Generate Rayleigh fading channel samples
            h_real = tf.random.normal([batch_size, self.system_params.num_rx, self.system_params.num_tx])
            h_imag = tf.random.normal([batch_size, self.system_params.num_rx, self.system_params.num_tx])
            h = tf.complex(h_real, h_imag)

            # Normalize channel matrix
            h_normalized = normalize_complex_tensor(h)

            # Apply path loss
            h_with_path_loss = h_normalized * tf.sqrt(tf.reshape(path_loss_linear, [-1, 1, 1]))

            # Add AWGN based on SNR
            noise_power = tf.pow(10.0, -snr_db / 10.0)
            noise_real = tf.random.normal(tf.shape(h_with_path_loss), mean=0.0, stddev=tf.sqrt(noise_power / 2.0))
            noise_imag = tf.random.normal(tf.shape(h_with_path_loss), mean=0.0, stddev=tf.sqrt(noise_power / 2.0))
            noise = tf.complex(noise_real, noise_imag)
            noisy_channel = h_with_path_loss + noise

            return {
                "perfect_channel": h_with_path_loss,
                "noisy_channel": noisy_channel,
            }

        except Exception as e:
            self.logger.error(f"Error generating channel samples: {str(e)}")
            raise

# Example usage
def main():
    system_params = SystemParameters()
    channel_manager = ChannelModelManager(system_params)

    # Generate channel samples
    batch_size = 32
    snr_db = tf.constant([20.0] * batch_size, dtype=tf.float32)
    channels = channel_manager.generate_channel_samples(batch_size, snr_db)

    print("Generated Channels:")
    for key, value in channels.items():
        print(f"{key}: shape={value.shape}")

if __name__ == "__main__":
    main()