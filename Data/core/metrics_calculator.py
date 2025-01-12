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

    def validate_performance_metrics(self, metrics: Dict[str, tf.Tensor]) -> Dict[str, bool]:
        """
        Validates if the calculated metrics meet the target requirements.
        
        Args:
            metrics: Dictionary containing calculated performance metrics
            
        Returns:
            Dictionary with validation results for each metric
        """
        validation_results = {}
        
        # Validate BER
        ber = metrics.get('ber', None)
        if ber is not None:
            # Check BER at 15dB SNR
            snr_15db_mask = tf.abs(metrics['snr_db'] - 15.0) < 0.5
            ber_at_15db = tf.boolean_mask(ber, snr_15db_mask)
            validation_results['ber_valid'] = tf.reduce_mean(ber_at_15db) < self.system_params.ber_target
            
        # Validate SINR
        sinr = metrics.get('sinr_db', None)
        if sinr is not None:
            validation_results['sinr_valid'] = tf.reduce_mean(sinr) > self.system_params.sinr_target
            
        # Validate Spectral Efficiency
        spectral_eff = metrics.get('spectral_efficiency', None)
        if spectral_eff is not None:
            se_valid = tf.logical_and(
                tf.reduce_mean(spectral_eff) >= self.system_params.spectral_efficiency_min,
                tf.reduce_mean(spectral_eff) <= self.system_params.spectral_efficiency_max
            )
            validation_results['spectral_efficiency_valid'] = se_valid
            
        return validation_results

    def log_validation_results(self, validation_results: Dict[str, bool]):
        """
        Logs the validation results with appropriate messages.
        """
        for metric, is_valid in validation_results.items():
            status = "PASSED" if is_valid else "FAILED"
            self.logger.info(f"Performance Validation - {metric}: {status}")
            
        if not all(validation_results.values()):
            self.logger.warning("Some performance targets were not met!")

    def calculate_performance_metrics(
        self,
        channel_response: tf.Tensor,
        tx_symbols: tf.Tensor,
        rx_symbols: tf.Tensor,
        snr_db: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Calculate base performance metrics for MIMO system.
        """
        # Ensure consistent dtypes
        channel_response = tf.cast(channel_response, tf.complex64)
        tx_symbols = tf.cast(tx_symbols, tf.complex64)
        rx_symbols = tf.cast(rx_symbols, tf.complex64)
        snr_db = tf.cast(snr_db, tf.float32)

        # Calculate MIMO metrics
        mimo_metrics = self.calculate_mimo_metrics(channel_response, snr_db)

        # Calculate signal power (using complex magnitude)
        signal_power = tf.reduce_mean(tf.abs(tx_symbols) ** 2, axis=-1)
        signal_power = tf.cast(signal_power, tf.float32)  # Convert to float32 for calculations
        signal_power = tf.clip_by_value(
            signal_power,
            self.validation_thresholds['signal_power']['min'],
            self.validation_thresholds['signal_power']['max']
        )

        # Calculate noise power (using complex difference)
        noise = rx_symbols - tx_symbols
        noise_power = tf.reduce_mean(tf.abs(noise) ** 2, axis=-1)
        noise_power = tf.cast(noise_power, tf.float32)  # Convert to float32 for calculations
        noise_power = tf.clip_by_value(
            noise_power,
            self.validation_thresholds['noise_power']['min'],
            self.validation_thresholds['noise_power']['max']
        )

        # Calculate SINR
        sinr = signal_power / (noise_power + 1e-10)
        sinr_db = 10.0 * tf.math.log(sinr) / tf.math.log(10.0)
        sinr_db = tf.clip_by_value(
            sinr_db,
            self.validation_thresholds['sinr']['min'],
            self.validation_thresholds['sinr']['max']
        )

        metrics = {
            'sinr_db': sinr_db,
            'signal_power': signal_power,
            'noise_power': noise_power,
            **mimo_metrics
        }

        return metrics

    def calculate_ber(self, tx_symbols: tf.Tensor, rx_symbols: tf.Tensor) -> tf.Tensor:
        """
        Calculate Bit Error Rate between transmitted and received symbols.
        
        Args:
            tx_symbols: Transmitted symbols [batch_size, num_streams]
            rx_symbols: Received symbols [batch_size, num_streams]
            
        Returns:
            BER values [batch_size]
        """
        # Convert symbols to bits (assuming QPSK modulation)
        tx_bits = tf.cast(tf.real(tx_symbols) > 0, tf.int32)
        rx_bits = tf.cast(tf.real(rx_symbols) > 0, tf.int32)
        
        # Calculate BER
        errors = tf.cast(tx_bits != rx_bits, tf.float32)
        ber = tf.reduce_mean(errors, axis=-1)
        
        return ber

    def calculate_enhanced_metrics(
        self,
        channel_response: tf.Tensor,
        tx_symbols: tf.Tensor,
        rx_symbols: tf.Tensor,
        snr_db: tf.Tensor
    ) -> Dict[str, Any]:
        """
        Calculate and validate enhanced performance metrics.
        
        Args:
            channel_response: Complex channel matrix [batch_size, num_rx, num_tx]
            tx_symbols: Transmitted symbols [batch_size, num_streams]
            rx_symbols: Received symbols [batch_size, num_streams]
            snr_db: SNR values in dB [batch_size]
            
        Returns:
            Dictionary containing calculated metrics and validation results
        """
        # Calculate base metrics
        metrics = self.calculate_performance_metrics(
            channel_response, tx_symbols, rx_symbols, snr_db
        )
        
        # Calculate BER
        ber = self.calculate_ber(tx_symbols, rx_symbols)
        metrics['ber'] = ber
        
        # Validate metrics against targets
        validation_results = self.validate_performance_metrics(metrics)
        metrics['validation_results'] = validation_results
        
        # Log validation results
        self.log_validation_results(validation_results)
        
        return metrics

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
        try:
            # Type validation and casting
            self.validate_tensor_types(channel_response)
            snr_db = tf.cast(snr_db, tf.float32)
            H = tf.cast(channel_response, tf.complex64)
            
            # Validate input shapes
            batch_size = tf.shape(H)[0]
            validate_mimo_tensor_shapes(
                channel_response=H,
                num_tx_antennas=self.system_params.num_tx_antennas,
                num_rx_antennas=self.system_params.num_rx_antennas,
                batch_size=batch_size
            )
            
            # Calculate channel matrix properties
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
            
            # Calculate eigenvalues and convert to float32
            eigenvalues = tf.cast(tf.abs(tf.linalg.eigvalsh(HH_stable)), tf.float32)
            epsilon_float = tf.cast(1e-10, tf.float32)
            
            # Use float32 for all subsequent calculations
            eigenvalues = tf.maximum(eigenvalues, epsilon_float)
            eigenvalues = eigenvalues / tf.reduce_max(eigenvalues, axis=1, keepdims=True)
            
            # Process SNR with validation
            snr_db = tf.clip_by_value(
                snr_db,
                self.validation_thresholds['sinr']['min'],
                self.validation_thresholds['sinr']['max']
            )
            snr_linear = tf.pow(10.0, snr_db/10.0)
            
            # Calculate spectral efficiency with validation
            spectral_efficiency = tf.reduce_sum(
                tf.math.log(1.0 + eigenvalues * tf.reshape(snr_linear, [-1, 1])) / tf.math.log(2.0),
                axis=1
            )
            spectral_efficiency = tf.clip_by_value(
                spectral_efficiency,
                self.validation_thresholds['spectral_efficiency']['min'],
                self.validation_thresholds['spectral_efficiency']['max']
            )
            
            # Calculate effective SNR with validation
            effective_snr = tf.reduce_mean(eigenvalues, axis=1) * snr_linear
            effective_snr = tf.maximum(effective_snr, self.validation_thresholds['signal_power']['min'])
            effective_snr_db = 10.0 * tf.math.log(effective_snr) / tf.math.log(10.0)
            effective_snr_db = tf.clip_by_value(
                effective_snr_db,
                self.validation_thresholds['effective_snr']['min'],
                self.validation_thresholds['effective_snr']['max']
            )
            
            # Calculate condition number with stability
            condition_number = tf.reduce_max(eigenvalues, axis=1) / tf.maximum(
                tf.reduce_min(eigenvalues, axis=1),
                epsilon_float
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