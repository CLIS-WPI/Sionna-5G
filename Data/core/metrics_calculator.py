# Performance metrics calculation
# core/metrics_calculator.py
# MIMO Communication System Performance Metrics Calculator
# Computes comprehensive performance metrics including SINR, spectral efficiency, and BER
# Provides advanced signal processing and statistical analysis for communication system evaluation

import tensorflow as tf
import numpy as np
from sionna.utils import compute_ber
from typing import Dict, Any, Optional

from config.system_parameters import SystemParameters
from utill.tensor_shape_validator import assert_tensor_shape

class MetricsCalculator:
    """
    Advanced performance metrics calculator for MIMO communication systems
    """
    
    def __init__(
        self, 
        system_params: Optional[SystemParameters] = None
    ):
        """
        Initialize metrics calculator with system parameters
        
        Args:
            system_params (Optional[SystemParameters]): System configuration
        """
        self.system_params = system_params or SystemParameters()
    
    def calculate_performance_metrics(
        self, 
        channel_response: tf.Tensor, 
        tx_symbols: tf.Tensor, 
        rx_symbols: tf.Tensor, 
        snr_db: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Calculate comprehensive MIMO performance metrics
        
        Args:
            channel_response (tf.Tensor): MIMO channel matrix
            tx_symbols (tf.Tensor): Transmitted symbols
            rx_symbols (tf.Tensor): Received symbols
            snr_db (tf.Tensor): Signal-to-Noise Ratio in dB
        
        Returns:
            Dict[str, tf.Tensor]: Performance metrics
        """
        # Validate input tensor shapes
        batch_size = tf.shape(channel_response)[0]
        
        assert_tensor_shape(
            channel_response, 
            [batch_size, self.system_params.num_rx, self.system_params.num_tx], 
            'channel_response'
        )
        assert_tensor_shape(
            tx_symbols, 
            [batch_size, self.system_params.num_tx, 1], 
            'tx_symbols'
        )
        assert_tensor_shape(
            rx_symbols, 
            [batch_size, self.system_params.num_rx, 1], 
            'rx_symbols'
        )
        assert_tensor_shape(snr_db, [batch_size], 'snr_db')
        
        # Convert SNR to linear scale
        snr_linear = tf.pow(10.0, snr_db / 10.0)
        snr_linear = tf.reshape(snr_linear, [batch_size, 1])
        
        # Cast channel response to complex64
        H = tf.cast(channel_response, tf.complex64)
        
        # Calculate channel eigenvalues
        H_conj_transpose = tf.linalg.adjoint(H)
        HH = tf.matmul(H_conj_transpose, H)
        eigenvalues = tf.abs(tf.linalg.eigvalsh(HH))
        
        # Signal calculation
        signal = tf.matmul(H, tx_symbols)
        
        # Signal power calculation
        signal_power = tf.reduce_mean(tf.abs(signal) ** 2, axis=[1, 2])
        
        # Noise power calculation
        noise_power = 1.0 / tf.squeeze(snr_linear)
        
        # SINR calculation
        sinr = signal_power / noise_power
        sinr = tf.expand_dims(sinr, axis=-1)
        sinr_db = 10.0 * tf.math.log(sinr) / tf.math.log(10.0)
        
        # Effective SNR calculation
        snr_linear_expanded = tf.expand_dims(snr_linear, axis=1)
        eigenvalues_snr = eigenvalues * snr_linear_expanded
        effective_snr = tf.reduce_mean(eigenvalues_snr, axis=1)
        effective_snr_db = 10.0 * tf.math.log(effective_snr) / tf.math.log(10.0)
        
        # Spectral efficiency calculation
        spectral_efficiency = tf.reduce_sum(
            tf.math.log(1.0 + eigenvalues * snr_linear_expanded) / tf.math.log(2.0),
            axis=1
        )
        
        return {
            'sinr': tf.squeeze(sinr_db),
            'spectral_efficiency': spectral_efficiency,
            'effective_snr': effective_snr_db,
            'eigenvalues': eigenvalues
        }
    
    def calculate_enhanced_metrics(
        self, 
        channel_response: tf.Tensor, 
        tx_symbols: tf.Tensor, 
        rx_symbols: tf.Tensor, 
        snr_db: tf.Tensor
    ) -> Dict[str, Any]:
        """
        Calculate advanced metrics including Bit Error Rate (BER) and throughput
        
        Args:
            channel_response (tf.Tensor): MIMO channel matrix
            tx_symbols (tf.Tensor): Transmitted symbols
            rx_symbols (tf.Tensor): Received symbols
            snr_db (tf.Tensor): Signal-to-Noise Ratio in dB
        
        Returns:
            Dict[str, Any]: Enhanced performance metrics
        """
        # Calculate base performance metrics
        base_metrics = self.calculate_performance_metrics(
            channel_response, tx_symbols, rx_symbols, snr_db
        )
        
        # Generate tx bits (assuming QPSK modulation)
        batch_size = tf.shape(channel_response)[0]
        bits_per_symbol = 2  # QPSK
        total_bits = self.system_params.num_tx * bits_per_symbol
        tx_bits = tf.random.uniform(
            [batch_size, total_bits], 
            minval=0, 
            maxval=2, 
            dtype=tf.int32
        )
        
        # Process received symbols for BER calculation
        rx_real = tf.sign(tf.math.real(rx_symbols))
        rx_imag = tf.sign(tf.math.imag(rx_symbols))
        
        rx_bits_combined = tf.concat([
            tf.reshape(rx_real, [batch_size, -1]),
            tf.reshape(rx_imag, [batch_size, -1])
        ], axis=1)
        
        # Calculate Bit Error Rate (BER)
        ber = compute_ber(tx_bits, rx_bits_combined)
        
        # Calculate throughput
        bandwidth = (
            self.system_params.num_subcarriers * 
            self.system_params.subcarrier_spacing
        )
        spectral_efficiency = tf.reduce_mean(base_metrics['spectral_efficiency'])
        throughput = spectral_efficiency * bandwidth
        
        return {
            **base_metrics,
            'ber': ber,
            'throughput': throughput,
            'inference_time': 0.0  # Placeholder for future timing
        }
    
    def calculate_ber(
        self, 
        tx_bits: tf.Tensor, 
        rx_bits: tf.Tensor
    ) -> tf.Tensor:
        """
        Calculate Bit Error Rate
        
        Args:
            tx_bits (tf.Tensor): Transmitted bits
            rx_bits (tf.Tensor): Received bits
        
        Returns:
            tf.Tensor: Bit Error Rate
        """
        return compute_ber(tx_bits, rx_bits)

# Example usage
def main():
    # Create metrics calculator with default parameters
    metrics_calculator = MetricsCalculator()
    
    # Simulate channel and symbol data
    batch_size = 32
    channel_response = tf.random.normal([batch_size, 4, 4], dtype=tf.complex64)
    tx_symbols = tf.random.normal([batch_size, 4, 1], dtype=tf.complex64)
    rx_symbols = tf.random.normal([batch_size, 4, 1], dtype=tf.complex64)
    snr_db = tf.random.uniform([batch_size], -10, 30)
    
    # Calculate performance metrics
    metrics = metrics_calculator.calculate_enhanced_metrics(
        channel_response, tx_symbols, rx_symbols, snr_db
    )
    
    # Print metrics
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()