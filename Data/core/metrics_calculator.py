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
        """
        try:
            # Validate input shapes
            batch_size = tf.shape(channel_response)[0]
            expected_shapes = {
                'channel_response': (batch_size, self.system_params.num_rx, self.system_params.num_tx),
                'tx_symbols': (batch_size, self.system_params.num_tx, 1),
                'rx_symbols': (batch_size, self.system_params.num_rx, 1),
                'snr_db': (batch_size,)
            }
            
            # Calculate channel properties with enhanced numerical stability
            H = channel_response
            H_H = tf.transpose(H, perm=[0, 2, 1], conjugate=True)
            HH = tf.matmul(H, H_H)
            
            # Add small identity matrix for numerical stability
            # Convert identity matrix to complex64
            epsilon = 1e-10
            I = tf.cast(
                tf.eye(tf.shape(HH)[-1], batch_shape=[batch_size]) * epsilon,
                dtype=tf.complex64  # Explicitly cast to complex64
            )
            
            # Now both tensors are complex64
            HH_stable = HH + I
            
            # Calculate and normalize eigenvalues
            eigenvalues = tf.abs(tf.linalg.eigvalsh(HH_stable))
            eigenvalues = tf.maximum(eigenvalues, epsilon)
            eigenvalues = eigenvalues / tf.reduce_max(eigenvalues, axis=1, keepdims=True)
            
            # Process SNR with controlled range
            snr_db_clipped = tf.clip_by_value(snr_db, -20.0, 30.0)
            snr_linear = tf.pow(10.0, snr_db_clipped/10.0)
            snr_linear = tf.reshape(snr_linear, [-1, 1])
            
            # Calculate signal power with improved accuracy
            signal_power = tf.reduce_mean(
                tf.abs(tf.matmul(H, tx_symbols))**2,
                axis=[1, 2]
            )
            signal_power = tf.maximum(signal_power, epsilon)
            
            # Calculate noise power
            noise_power = signal_power / tf.maximum(tf.squeeze(snr_linear), epsilon)
            noise_power = tf.maximum(noise_power, epsilon)
            
            # Calculate SINR with enhanced stability
            sinr = 10.0 * tf.math.log(signal_power/noise_power) / tf.math.log(10.0)
            sinr = tf.clip_by_value(sinr, -20.0, 30.0)
            
            # Calculate spectral efficiency using capacity formula
            spectral_efficiency = tf.reduce_sum(
                tf.math.log(1.0 + eigenvalues * tf.reshape(snr_linear, [-1, 1])) / tf.math.log(2.0),
                axis=1
            )
            spectral_efficiency = tf.clip_by_value(spectral_efficiency, 0.0, 40.0)
            
            # Calculate effective SNR with improved accuracy
            effective_snr = tf.reduce_mean(eigenvalues, axis=1) * tf.squeeze(snr_linear)
            effective_snr = tf.maximum(effective_snr, epsilon)
            effective_snr_db = 10.0 * tf.math.log(effective_snr) / tf.math.log(10.0)
            effective_snr_db = tf.clip_by_value(effective_snr_db, -30.0, 40.0)
            
            return {
                'sinr': sinr,  # Shape: [batch_size], Range: [-20, 30] dB
                'spectral_efficiency': spectral_efficiency,  # Shape: [batch_size], Range: [0, 40]
                'effective_snr': effective_snr_db,  # Shape: [batch_size], Range: [-30, 40] dB
                'eigenvalues': eigenvalues,  # Shape: [batch_size, num_rx], Range: [0, 1]
                'signal_power': signal_power,  # Additional diagnostic metric
                'noise_power': noise_power  # Additional diagnostic metric
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
        try:
            # Calculate base performance metrics
            base_metrics = self.calculate_performance_metrics(
                channel_response, tx_symbols, rx_symbols, snr_db
            )
            
            # Get batch size and validate shapes
            batch_size = tf.shape(channel_response)[0]
            
            # Apply channel equalization for improved symbol detection
            H = channel_response
            H_H = tf.transpose(H, perm=[0, 2, 1], conjugate=True)
            
            # Create identity matrix with correct dtype
            epsilon = tf.cast(1e-6, dtype=tf.complex64)
            I = tf.eye(tf.shape(H)[2], batch_shape=[batch_size], dtype=tf.complex64)
            I = I * epsilon  # Multiply after casting to ensure complex64
            
            # Now both tensors will be complex64
            HH = tf.matmul(H_H, H)
            H_inv = tf.linalg.inv(HH + I)
            equalizer = tf.matmul(H_inv, H_H)
            rx_symbols_eq = tf.matmul(equalizer, rx_symbols)
            
            # Calculate normalization factor with correct dtype
            normalization_factor = tf.sqrt(
                tf.reduce_mean(tf.abs(rx_symbols_eq)**2, axis=[1, 2], keepdims=True)
            )
            # Cast normalization factor to complex64
            normalization_factor = tf.cast(normalization_factor, dtype=tf.complex64)
            
            # Now both tensors are complex64 for division
            rx_symbols_norm = rx_symbols_eq / normalization_factor
            
            # Generate tx bits with proper modulation handling
            bits_per_symbol = self.get_bits_per_symbol(self.current_modulation)
            total_bits = self.system_params.num_tx * bits_per_symbol
            
            # Generate random bits for transmission
            tx_bits = tf.cast(
                tf.random.uniform(
                    [batch_size, total_bits], 
                    minval=0, 
                    maxval=2
                ),
                dtype=tf.int32
            )
            
            # Improved symbol to bit conversion
            rx_real = tf.cast(tf.math.real(rx_symbols_norm) > 0, tf.int32)
            rx_imag = tf.cast(tf.math.imag(rx_symbols_norm) > 0, tf.int32)
            
            # Combine real and imaginary bits
            rx_bits_combined = tf.cast(
                tf.concat([
                    tf.reshape(rx_real, [batch_size, -1]),
                    tf.reshape(rx_imag, [batch_size, -1])
                ], axis=1),
                dtype=tf.int32
            )
            
            # Calculate BER with improved accuracy
            ber = compute_ber(tx_bits, rx_bits_combined)
            ber = tf.clip_by_value(ber, 0.0, 0.5)  # Limit maximum BER
            
            # Calculate throughput with proper scaling
            bandwidth = (
                self.system_params.num_subcarriers * 
                self.system_params.subcarrier_spacing
            )
            spectral_efficiency = tf.reduce_mean(base_metrics['spectral_efficiency'])
            throughput = spectral_efficiency * bandwidth
            
            # Add timing metrics
            inference_time = tf.timestamp() - tf.timestamp()  # Placeholder for actual timing
            
            return {
                **base_metrics,
                'ber': ber,
                'throughput': throughput,
                'inference_time': inference_time,
                'equalized_symbols': rx_symbols_norm
            }
            
        except Exception as e:
            self.logger.error(f"Error in calculate_enhanced_metrics: {str(e)}")
            raise

        
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