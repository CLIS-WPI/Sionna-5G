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
from utill.logging_config import LoggerManager, create_ber_logger
from config.system_parameters import SystemParameters
from utill.tensor_shape_validator import validate_mimo_tensor_shapes


class MetricsCalculator:
    """Metrics calculator for MIMO dataset generation"""
    
    def __init__(self, system_params: Optional[SystemParameters] = None):
        """Initialize metrics calculator"""
        self.system_params = system_params or SystemParameters()
        self.logger = LoggerManager.get_logger(__name__)
        self.ber_logger = create_ber_logger()
        
        # Define validation thresholds
        self.validation_thresholds = {
            'sinr': {'min': -20.0, 'max': 30.0},
            'spectral_efficiency': {'min': 0.0, 'max': 40.0},
            'effective_snr': {'min': -30.0, 'max': 40.0},
            'signal_power': {'min': 1e-10, 'max': 1e3},
            'noise_power': {'min': 1e-10, 'max': 1e3}
        }

    def validate_tensor_types(self, channel_response: tf.Tensor) -> None:
        """Validate tensor types before computation"""
        if channel_response.dtype != tf.complex64:
            raise TypeError(f"Channel response must be complex64, got {channel_response.dtype}")
        
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

    def calculate_ber(self, tx_symbols: tf.Tensor, rx_symbols: tf.Tensor, 
        snr_db: tf.Tensor, modulation: str = "QPSK") -> Dict[str, tf.Tensor]:
        """Calculate BER using Sionna's built-in metrics"""
        try:
            import sionna as sn
            
            # Define modulation-specific parameters
            modulation_params = {
                "QPSK": {
                    "bits_per_symbol": 2,
                    "constellation_type": "qam",
                    "num_bits_per_symbol": 2,
                    "target_ber": 1e-5
                },
                "16QAM": {
                    "bits_per_symbol": 4,
                    "constellation_type": "qam",
                    "num_bits_per_symbol": 4,
                    "target_ber": 1e-4
                },
                "64QAM": {
                    "bits_per_symbol": 6,
                    "constellation_type": "qam",
                    "num_bits_per_symbol": 6,
                    "target_ber": 1e-3
                }
            }
            
            # Get modulation parameters
            mod_params = modulation_params[modulation]
            
            # Create mapper with correct constellation parameters
            mapper = sn.mapping.Mapper(
                constellation_type=mod_params["constellation_type"],
                num_bits_per_symbol=mod_params["num_bits_per_symbol"]
            )
            
            # Cast inputs
            tx_symbols = tf.cast(tx_symbols, tf.complex64)
            rx_symbols = tf.cast(rx_symbols, tf.complex64)
            
            # Demap received symbols to LLRs
            llr = mapper.demap(rx_symbols)
            
            # Get transmitted bits from symbols using the same mapper
            tx_bits = mapper.get_bits(tx_symbols)
            
            # Hard decisions on LLRs
            detected_bits = tf.cast(llr > 0, tf.int32)
            
            # Calculate average BER
            ber = sn.utils.count_errors(detected_bits, tx_bits) / tf.size(tx_bits, out_type=tf.float32)
            
            # Calculate BER curve for different SNR points
            snr_points = tf.range(15, 31, delta=2, dtype=tf.float32)  # Start from 15dB
            ber_curve = {}
            
            # Initialize target_met as False
            target_met = False
            
            for snr in snr_points:
                mask = tf.abs(snr_db - snr) < 0.5
                if tf.reduce_any(mask):
                    tx_bits_at_snr = tf.boolean_mask(tx_bits, mask)
                    detected_bits_at_snr = tf.boolean_mask(detected_bits, mask)
                    ber_at_snr = sn.utils.count_errors(detected_bits_at_snr, tx_bits_at_snr) / \
                                tf.size(tx_bits_at_snr, out_type=tf.float32)
                    ber_curve[float(snr)] = float(ber_at_snr)
                    
                    # Check if this is the target SNR (15dB) and update target_met
                    if abs(float(snr) - 15.0) < 0.5:
                        target_met = float(ber_at_snr) < mod_params["target_ber"]
                    
                    # Log BER measurements
                    self.logger.info(f"BER at {snr}dB: {ber_at_snr:.2e}")
                    if hasattr(self, 'ber_logger'):
                        self.ber_logger.info(
                            "BER Measurement",
                            extra={
                                'snr': float(snr),
                                'ber': float(ber_at_snr),
                                'modulation': modulation
                            }
                        )

            # If we didn't find any samples at 15dB SNR, target_met remains False
            return {
                'average_ber': float(ber),
                'ber_curve': ber_curve,
                'all_targets_met': target_met,
                'modulation': modulation,
                'bits_per_symbol': mod_params["bits_per_symbol"],
                'target_ber': mod_params["target_ber"]
            }
            
        except Exception as e:
            self.logger.error(f"Error in BER calculation: {str(e)}")
            raise
        
    def calculate_enhanced_metrics(
        self, 
        channel_response: tf.Tensor,
        tx_symbols: tf.Tensor,
        rx_symbols: tf.Tensor,
        snr_db: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """Calculate enhanced metrics with proper dtype and shape handling."""
        try:
            # Ensure proper dtypes
            channel_response = tf.cast(channel_response, tf.complex64)
            tx_symbols = tf.cast(tx_symbols, tf.complex64)
            rx_symbols = tf.cast(rx_symbols, tf.complex64)
            snr_db = tf.cast(snr_db, tf.float32)

            # Validate tensor types
            self.validate_tensor_types(channel_response)

            # Calculate singular values (ensure real output)
            s = tf.linalg.svd(channel_response, compute_uv=False)
            
            # Calculate condition number (real-valued)
            condition_number = tf.cast(
                tf.reduce_max(s, axis=-1) / tf.reduce_min(s, axis=-1),
                tf.float32
            )

            # Calculate channel capacity (real-valued)
            # Reshape snr_linear to match eigenvalues shape
            snr_linear = tf.pow(10.0, snr_db/10.0)
            snr_linear_expanded = tf.expand_dims(snr_linear, axis=-1)  # [batch_size, 1]
            
            capacity = tf.reduce_sum(
                tf.math.log(1.0 + snr_linear_expanded * tf.square(tf.abs(s))) / tf.math.log(2.0),
                axis=-1
            )

            # Calculate effective SNR (real-valued)
            channel_power = tf.reduce_mean(tf.square(tf.abs(channel_response)), axis=[-2, -1])
            
            # Reshape tx_symbols for matrix multiplication
            tx_symbols_reshaped = tf.expand_dims(tx_symbols, -1)  # [batch_size, num_streams, 1]
            
            # Calculate received signal without noise
            expected_rx = tf.matmul(channel_response, tx_symbols_reshaped)[:, :, 0]  # Remove last dimension
            
            # Calculate noise power
            noise_power = tf.reduce_mean(tf.square(tf.abs(rx_symbols - expected_rx)), axis=-1)
            
            # Calculate effective SNR
            effective_snr = tf.cast(
                10.0 * tf.math.log(channel_power / (noise_power + 1e-10)) / tf.math.log(10.0),
                tf.float32
            )

            # Calculate spectral efficiency (real-valued)
            spectral_efficiency = tf.cast(
                capacity / self.system_params.num_tx_antennas,
                tf.float32
            )

            return {
                'condition_number': condition_number,
                'effective_snr': effective_snr,
                'spectral_efficiency': spectral_efficiency
            }

        except Exception as e:
            self.logger.error(f"Error calculating enhanced metrics: {str(e)}")
            raise

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
        
        
    def calculate_mimo_metrics(
        self,
        channel_response: tf.Tensor,
        snr_db: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Calculate essential MIMO metrics for dataset generation with proper dtype handling
        
        Args:
            channel_response: Complex channel matrix [batch_size, num_rx, num_tx]
            snr_db: SNR values in dB [batch_size]
            
        Returns:
            Dictionary containing calculated metrics with consistent dtypes
        """
        try:
            # Ensure proper dtype casting at input
            channel_response = tf.cast(channel_response, tf.complex64)
            snr_db = tf.cast(snr_db, tf.float32)
            
            # Type validation
            if channel_response.dtype != tf.complex64:
                raise TypeError(f"Channel response must be complex64, got {channel_response.dtype}")
            
            # Get batch size and validate shapes
            batch_size = tf.shape(channel_response)[0]
            validate_mimo_tensor_shapes(
                channel_response=channel_response,
                num_tx_antennas=self.system_params.num_tx_antennas,
                num_rx_antennas=self.system_params.num_rx_antennas,
                batch_size=batch_size
            )
            
            # Calculate Hermitian transpose and matrix product
            H_H = tf.transpose(channel_response, perm=[0, 2, 1], conjugate=True)
            HH = tf.matmul(channel_response, H_H)
            
            # Add stability term for matrix operations
            epsilon_complex = tf.cast(1e-10, tf.complex64)
            I = tf.eye(
                tf.shape(HH)[-1],
                batch_shape=[batch_size],
                dtype=tf.complex64
            ) * epsilon_complex
            
            # Stabilized matrix for eigenvalue computation
            HH_stable = HH + I
            
            # Calculate eigenvalues with proper casting
            eigenvalues = tf.cast(
                tf.abs(tf.linalg.eigvalsh(HH_stable)),
                tf.float32
            )
            
            # Ensure numerical stability for float operations
            epsilon_float = tf.cast(1e-10, tf.float32)
            eigenvalues = tf.maximum(eigenvalues, epsilon_float)
            
            # Normalize eigenvalues
            max_eigenvalues = tf.reduce_max(eigenvalues, axis=1, keepdims=True)
            eigenvalues = eigenvalues / tf.maximum(max_eigenvalues, epsilon_float)
            
            # Process SNR with validation and proper casting
            snr_db = tf.clip_by_value(
                snr_db,
                self.validation_thresholds['sinr']['min'],
                self.validation_thresholds['sinr']['max']
            )
            snr_linear = tf.cast(tf.pow(10.0, snr_db/10.0), tf.float32)
            
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
            
            # Calculate effective SNR with proper casting
            effective_snr = tf.cast(
                tf.reduce_mean(eigenvalues, axis=1) * snr_linear,
                tf.float32
            )
            effective_snr = tf.maximum(
                effective_snr,
                self.validation_thresholds['signal_power']['min']
            )
            
            # Convert to dB with proper casting
            effective_snr_db = tf.cast(
                10.0 * tf.math.log(effective_snr) / tf.math.log(10.0),
                tf.float32
            )
            effective_snr_db = tf.clip_by_value(
                effective_snr_db,
                self.validation_thresholds['effective_snr']['min'],
                self.validation_thresholds['effective_snr']['max']
            )
            
            # Calculate condition number with stability
            condition_number = tf.cast(
                tf.reduce_max(eigenvalues, axis=1) / tf.maximum(
                    tf.reduce_min(eigenvalues, axis=1),
                    epsilon_float
                ),
                tf.float32
            )
            
            # Return metrics with consistent dtypes
            return {
                'spectral_efficiency': tf.cast(spectral_efficiency, tf.float32),
                'effective_snr': tf.cast(effective_snr_db, tf.float32),
                'eigenvalues': tf.cast(eigenvalues, tf.float32),
                'condition_number': tf.cast(condition_number, tf.float32)
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