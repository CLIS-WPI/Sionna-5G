#class ChannelModelManager
# core/channel_model.py
# MIMO Channel Model Generation and Management
# Handles complex channel sample generation, antenna array configuration, and resource grid setup
# Provides advanced channel modeling capabilities with flexible system parameter integration

import tensorflow as tf
import numpy as np
from sionna.channel import RayleighBlockFading
from sionna.channel.tr38901 import AntennaArray
from sionna.ofdm import ResourceGrid
from typing import Tuple, Optional, Dict
from sionna.mapping import Mapper
from config.system_parameters import SystemParameters
from utill.tensor_shape_validator import assert_tensor_shape, normalize_complex_tensor
from sionna.mimo import StreamManagement
from utill.logging_config import LoggerManager

class ChannelModelManager:
    """
    Advanced channel model management for MIMO communication systems
    """
    
    def __init__(self, system_params: Optional[SystemParameters] = None):
        # Use default parameters if not provided
        self.system_params = system_params or SystemParameters()
        self.logger = LoggerManager.get_logger(__name__)
    
        # Initialize logger
        self.logger = LoggerManager.get_logger(__name__)

        # Initialize mapper with correct number of bits (default to QPSK)
        self.mapper = Mapper("qam", num_bits_per_symbol=2)

        # Initialize stream management with correct parameters
        # Create rx_tx_association matrix: each receiver is associated with transmitter 0
        rx_tx_association = [[0] for _ in range(self.system_params.num_rx)]

        # Initialize stream management with correct parameters
        self.stream_management = StreamManagement(
            num_streams_per_tx=1,  # Single integer value instead of a list
            rx_tx_association=rx_tx_association  # Required parameter showing which RX is associated with which TX
        )

        # Initialize channel components
        self._setup_antenna_arrays()
        self._setup_resource_grid()
        self._setup_channel_model()

    def generate_qam_symbols(self, batch_size: int, mod_scheme: str) -> tf.Tensor:
        try:
            # Validate input parameters
            if batch_size <= 0:
                raise ValueError("Batch size must be positive")
                
            # Enhanced modulation scheme handling
            constellation_size = {
                'BPSK': 2,
                'QPSK': 4,
                '16QAM': 16,
                '64QAM': 64,
                '256QAM': 256
            }.get(mod_scheme.upper())
            
            if constellation_size is None:
                self.logger.warning(f"Unknown modulation scheme {mod_scheme}, defaulting to QPSK")
                constellation_size = 4
                
            # Calculate bits per symbol
            num_bits_per_symbol = int(np.log2(constellation_size))
            total_bits = batch_size * self.system_params.num_tx * num_bits_per_symbol
            
            # Create mapper with proper configuration
            mapper = Mapper("qam", num_bits_per_symbol=num_bits_per_symbol)
            
            # Generate random bits
            bits = tf.random.uniform(
                [total_bits], 
                minval=0, 
                maxval=2, 
                dtype=tf.int32
            )
            
            # Reshape bits for mapping
            bits = tf.reshape(bits, [batch_size, self.system_params.num_tx, num_bits_per_symbol])
            
            # Map bits to QAM symbols
            symbols = mapper(bits)
            
            # Normalize symbols
            symbols = normalize_complex_tensor(symbols)
            
            return tf.reshape(symbols, [batch_size, self.system_params.num_tx, 1])
            
        except Exception as e:
            self.logger.error(f"Error generating QAM symbols: {str(e)}")
            raise
        
    def _setup_antenna_arrays(self) -> None:
        """
        Configure antenna arrays for transmitter and receiver
        """
        # Transmitter array configuration
        self.tx_array = AntennaArray(
            num_rows=self.system_params.num_tx,            
            num_cols=1,                      
            polarization="dual",           
            polarization_type="VH",        
            antenna_pattern="38.901",     
            carrier_frequency=self.system_params.carrier_frequency,
            dtype=tf.complex64
        )
        
        # Receiver array configuration
        self.rx_array = AntennaArray(
            num_rows=self.system_params.num_rx,            
            num_cols=1,                      
            polarization="single",           
            polarization_type="V",           
            antenna_pattern="omni",          
            carrier_frequency=self.system_params.carrier_frequency,
            dtype=tf.complex64
        )
    
    def _setup_resource_grid(self) -> None:
        """
        Configure OFDM resource grid
        """
        # Configure resource grid with system parameters
        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=self.system_params.num_ofdm_symbols,
            fft_size=self.system_params.num_subcarriers,
            subcarrier_spacing=self.system_params.subcarrier_spacing,
            num_tx=self.system_params.num_tx,
            num_streams_per_tx=1,
            cyclic_prefix_length=16,
            num_guard_carriers=(6, 6),
            dc_null=False,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11],
            dtype=tf.complex64
        )
    
    def _setup_channel_model(self) -> None:
        """
        Initialize Rayleigh block fading channel model
        """
        self.channel_model = RayleighBlockFading(
            num_rx=1,
            num_rx_ant=self.system_params.num_rx,
            num_tx=1,
            num_tx_ant=self.system_params.num_tx,
            dtype=tf.complex64
        )
    
    
    def generate_channel_samples(self, batch_size: int, snr_db: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        try:
            # Cast batch_size to int32 if needed
            batch_size = tf.cast(batch_size, tf.int32)
            
            # Generate random channel matrix
            # The channel_model should return a complex tensor
            h_raw = self.channel_model(batch_size, num_time_steps=1)
            
            # Ensure we have a complex64 tensor before any operations
            h_complex = tf.dtypes.cast(h_raw, tf.complex64)
            
            # Remove the time dimension safely
            h = tf.squeeze(h_complex, axis=1)
            
            # Validate tensor shapes
            h = assert_tensor_shape(
                h, 
                [batch_size, self.system_params.num_rx, self.system_params.num_tx], 
                'channel_matrix'
            )
            
            # Normalize channel power
            h_normalized = normalize_complex_tensor(h)
            
            # Calculate noise power
            noise_power = tf.cast(1.0 / tf.pow(10.0, snr_db / 10.0), dtype=tf.float32)
            noise_power = tf.maximum(noise_power, 1e-10)
            noise_power = tf.reshape(noise_power, [-1, 1, 1])
            
            # Generate complex Gaussian noise
            std_dev = tf.sqrt(noise_power / 2)
            noise = tf.complex(
                tf.random.normal(h_normalized.shape, mean=0.0, stddev=std_dev),
                tf.random.normal(h_normalized.shape, mean=0.0, stddev=std_dev)
            )
            
            return h_normalized, h_normalized + noise
            
        except Exception as e:
            self.logger.error(f"Error generating channel samples: {str(e)}")
            raise

    def generate_mimo_channel(
        self, 
        batch_size: int
    ) -> Dict[str, tf.Tensor]:
        """
        Generate comprehensive MIMO channel data
        
        Args:
            batch_size (int): Number of channel realizations
        
        Returns:
            Dict[str, tf.Tensor]: Dictionary of channel-related tensors
        """
        # Generate random SNR values within system-defined range
        snr_db = tf.random.uniform(
            [batch_size], 
            self.system_params.snr_range[0], 
            self.system_params.snr_range[1]
        )
        
        # Generate channel samples
        perfect_channel, noisy_channel = self.generate_channel_samples(
            batch_size, snr_db
        )
        
        return {
            'perfect_channel': perfect_channel,
            'noisy_channel': noisy_channel,
            'snr_db': snr_db
        }
    
    def get_channel_statistics(
        self, 
        channel: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Compute channel matrix statistics
        
        Args:
            channel (tf.Tensor): Input channel matrix
        
        Returns:
            Dict[str, tf.Tensor]: Channel statistical properties
        """
        return {
            'mean': tf.reduce_mean(channel),
            'variance': tf.math.reduce_variance(channel),
            'magnitude': tf.abs(channel),
            'power': tf.reduce_mean(tf.abs(channel) ** 2)
        }
    def get_bits_per_symbol(self, mod_scheme: str) -> int:
        """
        Determine bits per symbol based on modulation scheme
        
        Args:
            mod_scheme (str): Modulation scheme
        
        Returns:
            int: Number of bits per symbol
        """
        modulation_bits = {
            'QPSK': 2,
            '16QAM': 4,
            '64QAM': 6
        }
        return modulation_bits.get(mod_scheme, 2)  # Default to QPSK if not found

    # Then in the methods where you generate symbols, you can use it like:
    def generate_symbols(self, mod_scheme: str, batch_size: int):
        bits_per_symbol = self.get_bits_per_symbol(mod_scheme)
        return self.resource_grid.qam_source(num_bits_per_symbol=bits_per_symbol)(
            [batch_size, self.system_params.num_tx, 1]
        )
    
# Example usage
def main():
    # Create channel model with default parameters
    channel_manager = ChannelModelManager()
    
    # Generate MIMO channel samples
    mimo_channel = channel_manager.generate_mimo_channel(batch_size=32)
    
    # Print channel statistics
    for key, value in mimo_channel.items():
        print(f"{key} shape: {value.shape}")

if __name__ == "__main__":
    main()