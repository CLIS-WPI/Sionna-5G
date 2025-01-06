# Performance metrics calculation
# core/metrics_calculator.py
# MIMO Communication System Performance Metrics Calculator
# Computes comprehensive performance metrics including SINR, spectral efficiency, and BER
# Provides advanced signal processing and statistical analysis for communication system evaluation
# Numerical Stability:
# - Inputs and outputs must be free of NaN, Infinity, or extreme outliers.
# - Replace invalid values with defaults and clip values to acceptable thresholds.

import tensorflow as tf
import numpy as np
from sionna.utils import compute_ber
from typing import Dict, List, Any, Optional
from utill.logging_config import LoggerManager
from config.system_parameters import SystemParameters
from utill.tensor_shape_validator import validate_mimo_tensor_shapes
import logging
class MetricsCalculator:
    """Metrics calculator for MIMO dataset generation"""
    
    def __init__(self, system_params: Optional[SystemParameters] = None):
        """Initialize metrics calculator"""
        self.system_params = system_params or SystemParameters()
        self.logger = LoggerManager.get_logger(__name__)
        
        # Define validation thresholds
        self.validation_thresholds = {
            'sinr': {'min': -20.0, 'max': 30.0},
            'spectral_efficiency': {'min': 0.0, 'max': 40.0},
            'effective_snr': {'min': -30.0, 'max': 40.0},
            'signal_power': {'min': 1e-10, 'max': 1e3},
            'noise_power': {'min': 1e-10, 'max': 1e3}
        }
    def assert_tensor_shape(tensor: tf.Tensor, expected_shape: List[int], name: str = "tensor") -> bool:
        """
        Assert that tensor has expected shape
        
        Args:
            tensor (tf.Tensor): Input tensor
            expected_shape (List[int]): Expected tensor shape
            name (str): Tensor name for error reporting
            
        Returns:
            bool: True if shape matches
        """
        try:
            actual_shape = tensor.shape.as_list()
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch for {name}: "
                    f"Expected {expected_shape}, got {actual_shape}"
                )
            return True
        except Exception as e:
            logging.error(f"Tensor shape assertion failed: {str(e)}")
            return False
        
    def validate_tensor_types(self, channel_response: tf.Tensor) -> None:
        """Validate tensor types before computation"""
        if channel_response.dtype != tf.complex64:
            raise TypeError(f"Channel response must be complex64, got {channel_response.dtype}")
        
    def calculate_mimo_metrics(
        self,
        channel_response: tf.Tensor,
        snr_db: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Calculate essential MIMO metrics for dataset generation
        
        Args:
            channel_response: Complex channel matrix [batch_size, num_rx, num_tx]
            snr_db: SNR values in dB [batch_size]
            
        Returns:
            Dictionary containing calculated metrics
        """
        self.validate_tensor_types(channel_response)
        # Ensure float32 for SNR calculations
        snr_db = tf.cast(snr_db, tf.float32)
        # Calculate channel matrix properties
        H = tf.cast(channel_response, tf.complex64)
        H_H = tf.transpose(H, perm=[0, 2, 1], conjugate=True)
        try:
            # Validate input shapes
            batch_size = tf.shape(channel_response)[0]
            expected_shape = [
                batch_size,
                self.system_params.num_rx_antennas,
                self.system_params.num_tx_antennas
            ]
            validate_mimo_tensor_shapes(
                channel_response=channel_response,
                num_tx_antennas=self.system_params.num_tx_antennas,
                num_rx_antennas=self.system_params.num_rx_antennas,
                batch_size=batch_size
            )
            
            # Calculate channel matrix properties
            H = channel_response
            H_H = tf.transpose(H, perm=[0, 2, 1], conjugate=True)
            HH = tf.matmul(H, H_H)
            
            # Add stability term
            epsilon = tf.cast(1e-10, dtype=tf.complex64)
            I = tf.eye(
                tf.shape(HH)[-1],
                batch_shape=[batch_size],
                dtype=tf.complex64
            ) * epsilon
            
            HH_stable = HH + I
            
            # Calculate eigenvalues
            eigenvalues = tf.abs(tf.linalg.eigvalsh(HH_stable))
            eigenvalues = tf.maximum(eigenvalues, epsilon)
            eigenvalues = eigenvalues / tf.reduce_max(eigenvalues, axis=1, keepdims=True)
            
            # Process SNR
            snr_db = tf.clip_by_value(
                snr_db,
                self.validation_thresholds['sinr']['min'],
                self.validation_thresholds['sinr']['max']
            )
            snr_linear = tf.pow(10.0, snr_db/10.0)
            
            # Calculate spectral efficiency
            spectral_efficiency = tf.reduce_sum(
                tf.math.log(1.0 + eigenvalues * tf.reshape(snr_linear, [-1, 1])) / tf.math.log(2.0),
                axis=1
            )
            spectral_efficiency = tf.clip_by_value(
                spectral_efficiency,
                self.validation_thresholds['spectral_efficiency']['min'],
                self.validation_thresholds['spectral_efficiency']['max']
            )
            
            # Calculate effective SNR
            effective_snr = tf.reduce_mean(eigenvalues, axis=1) * snr_linear
            effective_snr = tf.maximum(effective_snr, self.validation_thresholds['signal_power']['min'])
            effective_snr_db = 10.0 * tf.math.log(effective_snr) / tf.math.log(10.0)
            effective_snr_db = tf.clip_by_value(
                effective_snr_db,
                self.validation_thresholds['effective_snr']['min'],
                self.validation_thresholds['effective_snr']['max']
            )
            
            # Calculate condition number
            condition_number = tf.reduce_max(eigenvalues, axis=1) / tf.maximum(
                tf.reduce_min(eigenvalues, axis=1),
                epsilon
            )
            
            return {
                'spectral_efficiency': spectral_efficiency,
                'effective_snr': effective_snr_db,
                'eigenvalues': eigenvalues,
                'condition_number': condition_number
            }
            
        except Exception as e:
            self.logger.error(f"Error in calculate_mimo_metrics: {str(e)}")
            raise

def main():
    """Test the metrics calculator"""
    # Initialize calculator
    calc = MetricsCalculator()
    
    # Generate test data
    batch_size = 1000
    channel_response = tf.complex(
        tf.random.normal([batch_size, 4, 4]),
        tf.random.normal([batch_size, 4, 4])
    )
    snr_db = tf.random.uniform([batch_size], -10, 30)
    
    # Calculate metrics
    metrics = calc.calculate_mimo_metrics(channel_response, snr_db)
    
    # Print statistics
    for key, value in metrics.items():
        if isinstance(value, tf.Tensor):
            print(f"{key}:")
            print(f"  Mean: {tf.reduce_mean(value).numpy():.4f}")
            print(f"  Std:  {tf.math.reduce_std(value).numpy():.4f}")
            print(f"  Min:  {tf.reduce_min(value).numpy():.4f}")
            print(f"  Max:  {tf.reduce_max(value).numpy():.4f}")

if __name__ == "__main__":
    main()