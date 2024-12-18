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
    
    
    
    def generate_channel_samples(self, batch_size: int, snr_db: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Generate MIMO channel samples with controlled scaling and normalization
        
        Args:
            batch_size (int): Number of samples to generate
            snr_db (tf.Tensor): Signal-to-Noise Ratio in dB
            
        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Perfect channel, noisy channel, eigenvalues
        """
        try:
            batch_size = tf.cast(batch_size, tf.int32)
            h_shape = [batch_size, self.system_params.num_rx, self.system_params.num_tx]

            # Use controlled standard deviation for better stability
            std_dev = 1.0/np.sqrt(2.0 * self.system_params.num_tx)
            
            # Generate channel components with controlled variance
            h_real = tf.random.normal(h_shape, mean=0.0, stddev=std_dev)
            h_imag = tf.random.normal(h_shape, mean=0.0, stddev=std_dev)
            h = tf.complex(h_real, h_imag)
            
            # Validate and normalize channel matrix
            h = assert_tensor_shape(h, h_shape, 'channel_matrix')
            h_normalized = normalize_complex_tensor(h)
            
            # Calculate eigenvalues with proper normalization
            h_hermitian = tf.matmul(
                h_normalized, 
                tf.transpose(h_normalized, perm=[0, 2, 1], conjugate=True)
            )
            eigenvalues = tf.abs(tf.linalg.eigvals(h_hermitian))
            
            # Normalize eigenvalues to [0, 1] range
            eigenvalues = eigenvalues / tf.reduce_max(eigenvalues, axis=1, keepdims=True)
            
            # Clip SNR to valid range and calculate noise power
            snr_db_clipped = tf.clip_by_value(snr_db, -20.0, 30.0)
            noise_power = tf.pow(10.0, -snr_db_clipped / 10.0)
            noise_power = tf.maximum(noise_power, 1e-10)
            noise_power = tf.reshape(noise_power, [-1, 1, 1])
            
            # Generate noise with controlled variance
            noise_std = tf.sqrt(noise_power / 2.0)
            noise_real = tf.random.normal(tf.shape(h_normalized), mean=0.0, stddev=noise_std)
            noise_imag = tf.random.normal(tf.shape(h_normalized), mean=0.0, stddev=noise_std)
            noise = tf.complex(noise_real, noise_imag)
            
            # Add noise to channel
            noisy_channel = h_normalized + noise
            
            return h_normalized, noisy_channel, eigenvalues
                
        except Exception as e:
            self.logger.error(f"Error generating channel samples: {str(e)}")
            raise

    def calculate_effective_snr(self, channel_response: tf.Tensor, snr_db: tf.Tensor) -> tf.Tensor:
        """
        Calculate effective SNR based on channel response and nominal SNR
        """
        try:
            # Convert SNR to linear scale
            snr_linear = tf.pow(10.0, snr_db/10.0)
            
            # Calculate channel gain (Frobenius norm squared)
            channel_gain = tf.reduce_sum(
                tf.abs(channel_response)**2, 
                axis=[1, 2]
            ) / (self.system_params.num_rx * self.system_params.num_tx)
            
            # Calculate effective SNR
            effective_snr = channel_gain * snr_linear
            
            # Convert back to dB
            effective_snr_db = 10.0 * tf.math.log(effective_snr) / tf.math.log(10.0)
            
            # Clip values to reasonable range
            effective_snr_db = tf.clip_by_value(effective_snr_db, -30.0, 40.0)
            
            return effective_snr_db
            
        except Exception as e:
            self.logger.error(f"Error calculating effective SNR: {str(e)}")
            raise

    def calculate_spectral_efficiency(self, channel_response: tf.Tensor, snr_db: tf.Tensor) -> tf.Tensor:
        """
        Calculate spectral efficiency using the channel capacity formula
        """
        try:
            # Convert SNR to linear scale
            snr_linear = tf.pow(10.0, snr_db/10.0)
            snr_linear = tf.reshape(snr_linear, [-1, 1, 1])
            
            # Calculate channel correlation matrix
            h_hermitian = tf.matmul(
                channel_response, 
                tf.transpose(channel_response, perm=[0, 2, 1], conjugate=True)
            )
            
            # Calculate eigenvalues
            eigenvalues = tf.abs(tf.linalg.eigvals(h_hermitian))
            
            # Calculate capacity using eigenvalues
            capacity = tf.reduce_sum(
                tf.math.log(1.0 + snr_linear * eigenvalues) / tf.math.log(2.0),
                axis=1
            )
            
            # Convert to spectral efficiency (bits/s/Hz)
            spectral_efficiency = capacity / self.system_params.num_tx
            
            return spectral_efficiency
            
        except Exception as e:
            self.logger.error(f"Error calculating spectral efficiency: {str(e)}")
            raise
        
    def generate_mimo_channel(
        self, 
        batch_size: int,
        snr_db: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Generate comprehensive MIMO channel data
        
        Args:
            batch_size (int): Number of channel realizations
            snr_db (tf.Tensor): Signal-to-Noise Ratio in dB
        
        Returns:
            Dict[str, tf.Tensor]: Dictionary of channel-related tensors
        """
        try:
            # Generate channel samples - now correctly unpacking 3 values
            perfect_channel, noisy_channel, eigenvalues = self.generate_channel_samples(
                batch_size, snr_db
            )
            
            # Calculate additional metrics
            effective_snr = self.calculate_effective_snr(perfect_channel, snr_db)
            spectral_eff = self.calculate_spectral_efficiency(perfect_channel, snr_db)
            
            return {
                'perfect_channel': perfect_channel,
                'noisy_channel': noisy_channel,
                'eigenvalues': eigenvalues,
                'snr_db': snr_db,
                'effective_snr': effective_snr,
                'spectral_efficiency': spectral_eff
            }
            
        except Exception as e:
            self.logger.error(f"Error in generate_mimo_channel: {str(e)}")
            raise
    
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