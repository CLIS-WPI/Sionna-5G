# Performance metrics calculation
# core/metrics_calculator.py
# MIMO Communication System Performance Metrics Calculator
# Computes comprehensive performance metrics including SINR, spectral efficiency, and BER
# Provides advanced signal processing and statistical analysis for communication system evaluation

import tensorflow as tf
import numpy as np
from sionna.utils import compute_ber
from typing import Dict, Any, Optional
from utill.logging_config import LoggerManager
from config.system_parameters import SystemParameters
from utill.tensor_shape_validator import assert_tensor_shape
from utill.logging_config import LoggerManager  

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
        # Initialize system parameters
        self.system_params = system_params or SystemParameters()
        
        # Initialize logger
        self.logger = LoggerManager.get_logger(__name__)

    def calculate_performance_metrics(
        self, 
        channel_response: tf.Tensor, 
        tx_symbols: tf.Tensor, 
        rx_symbols: tf.Tensor, 
        snr_db: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Calculate comprehensive MIMO performance metrics with enhanced stability
        
        Args:
            channel_response (tf.Tensor): MIMO channel matrix [batch_size, num_rx, num_tx]
            tx_symbols (tf.Tensor): Transmitted symbols [batch_size, num_tx, 1]
            rx_symbols (tf.Tensor): Received symbols [batch_size, num_rx, 1]
            snr_db (tf.Tensor): Signal-to-Noise Ratio in dB [batch_size]
        
        Returns:
            Dict[str, tf.Tensor]: Performance metrics with controlled ranges
        """
        try:
            # Get batch size and validate shapes
            batch_size = tf.shape(channel_response)[0]
            
            # Calculate channel properties with enhanced numerical stability
            H_conj_transpose = tf.linalg.adjoint(channel_response)
            HH = tf.matmul(channel_response, H_conj_transpose)
            
            # Calculate and normalize eigenvalues
            eigenvalues = tf.abs(tf.linalg.eigvalsh(HH))
            eigenvalues = tf.maximum(eigenvalues, 1e-10)  # Prevent zero eigenvalues
            eigenvalues = eigenvalues / tf.reduce_max(eigenvalues, axis=1, keepdims=True)
            
            # Process SNR with clipping
            snr_db_clipped = tf.clip_by_value(snr_db, -20.0, 30.0)
            snr_linear = tf.pow(10.0, snr_db_clipped/10.0)
            snr_linear = tf.reshape(snr_linear, [-1, 1])
            
            # Calculate signal and noise power with stability checks
            signal_power = tf.maximum(
                tf.reduce_mean(tf.abs(rx_symbols)**2, axis=[1, 2]),
                1e-10
            )
            
            noise_power = tf.maximum(
                signal_power / tf.squeeze(snr_linear),
                1e-10
            )
            
            # Calculate SINR with controlled range
            sinr = 10.0 * tf.math.log(signal_power/noise_power) / tf.math.log(10.0)
            sinr = tf.clip_by_value(sinr, -30.0, 40.0)
            sinr = tf.reshape(sinr, [-1])
            
            # Calculate spectral efficiency with enhanced stability
            spectral_efficiency = tf.reduce_sum(
                tf.math.log(1.0 + eigenvalues * tf.reshape(snr_linear, [-1, 1])) / tf.math.log(2.0),
                axis=1
            )
            spectral_efficiency = tf.maximum(spectral_efficiency, 0.0)
            spectral_efficiency = tf.reshape(spectral_efficiency, [-1])
            
            # Calculate effective SNR with controlled range
            effective_snr = tf.reduce_mean(eigenvalues, axis=1) * tf.squeeze(snr_linear)
            effective_snr = tf.maximum(effective_snr, 1e-10)
            effective_snr_db = 10.0 * tf.math.log(effective_snr) / tf.math.log(10.0)
            effective_snr_db = tf.clip_by_value(effective_snr_db, -30.0, 40.0)
            effective_snr_db = tf.reshape(effective_snr_db, [-1])
            
            return {
                'sinr': sinr,  # Shape: [batch_size], Range: [-30, 40] dB
                'spectral_efficiency': spectral_efficiency,  # Shape: [batch_size], Range: [0, inf)
                'effective_snr': effective_snr_db,  # Shape: [batch_size], Range: [-30, 40] dB
                'eigenvalues': eigenvalues  # Shape: [batch_size, num_rx], Range: [0, 1]
            }
            
        except Exception as e:
            self.logger.error(f"Error in calculate_performance_metrics: {str(e)}")
            raise

    def calculate_enhanced_metrics(
        self, 
        channel_response: tf.Tensor, 
        tx_symbols: tf.Tensor, 
        rx_symbols: tf.Tensor, 
        snr_db: tf.Tensor
    ) -> Dict[str, Any]:
        """
        Calculate enhanced metrics including BER and throughput with proper shape handling
        """
        try:
            # Calculate base performance metrics
            base_metrics = self.calculate_performance_metrics(
                channel_response, tx_symbols, rx_symbols, snr_db
            )
            
            # Generate tx bits
            batch_size = tf.shape(channel_response)[0]
            bits_per_symbol = 2  # QPSK
            total_bits = self.system_params.num_tx * bits_per_symbol
            tx_bits = tf.cast(
                tf.random.uniform(
                    [batch_size, total_bits], 
                    minval=0, 
                    maxval=2
                ),
                dtype=tf.int32
            )
            
            # Process received symbols for BER calculation
            rx_real = tf.cast(tf.sign(tf.math.real(rx_symbols)) > 0, tf.int32)
            rx_imag = tf.cast(tf.sign(tf.math.imag(rx_symbols)) > 0, tf.int32)
            
            rx_bits_combined = tf.cast(
                tf.concat([
                    tf.reshape(rx_real, [batch_size, -1]),
                    tf.reshape(rx_imag, [batch_size, -1])
                ], axis=1),
                dtype=tf.int32
            )
            
            # Calculate BER
            ber = compute_ber(tx_bits, rx_bits_combined)
            ber = tf.reshape(ber, [-1])  # Ensure shape is [batch_size]
            
            # Calculate throughput
            bandwidth = self.system_params.num_subcarriers * self.system_params.subcarrier_spacing
            spectral_efficiency = tf.reduce_mean(base_metrics['spectral_efficiency'])
            throughput = spectral_efficiency * bandwidth
            throughput = tf.broadcast_to(throughput, [batch_size])  # Shape: [batch_size]
            
            return {
                'ber': ber,  # Shape: [batch_size]
                'throughput': throughput,  # Shape: [batch_size]
                'inference_time': tf.zeros([batch_size])  # Shape: [batch_size]
            }
            
        except Exception as e:
            self.logger.error(f"Error in calculate_enhanced_metrics: {str(e)}")
            raise

        
    def calculate_enhanced_metrics(
        self, 
        channel_response: tf.Tensor, 
        tx_symbols: tf.Tensor, 
        rx_symbols: tf.Tensor, 
        snr_db: tf.Tensor
    ) -> Dict[str, Any]:
        """
        Calculate advanced metrics including Bit Error Rate (BER) and throughput
        """
        # Calculate base performance metrics
        base_metrics = self.calculate_performance_metrics(
            channel_response, tx_symbols, rx_symbols, snr_db
        )
        
        # Generate tx bits (ensuring int32 type)
        batch_size = tf.shape(channel_response)[0]
        bits_per_symbol = 2  # QPSK
        total_bits = self.system_params.num_tx * bits_per_symbol
        tx_bits = tf.cast(
            tf.random.uniform(
                [batch_size, total_bits], 
                minval=0, 
                maxval=2
            ),
            dtype=tf.int32
        )
        
        # Process received symbols for BER calculation
        # Convert complex symbols to bits (ensuring int32 type)
        rx_real = tf.cast(tf.sign(tf.math.real(rx_symbols)) > 0, tf.int32)
        rx_imag = tf.cast(tf.sign(tf.math.imag(rx_symbols)) > 0, tf.int32)
        
        rx_bits_combined = tf.cast(
            tf.concat([
                tf.reshape(rx_real, [batch_size, -1]),
                tf.reshape(rx_imag, [batch_size, -1])
            ], axis=1),
            dtype=tf.int32
        )
        
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
            'inference_time': 0.0
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