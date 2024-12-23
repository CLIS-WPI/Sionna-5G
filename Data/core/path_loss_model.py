# Path loss calculations
# core/path_loss_model.py
# Advanced Path Loss Modeling and Calculation
# Implements multiple path loss scenarios with comprehensive statistical analysis
# Provides flexible path loss computation for different wireless communication environments

import tensorflow as tf
import numpy as np
from sionna.channel import tr38901
from typing import Optional, Tuple, Callable
from config.system_parameters import SystemParameters
from utill.tensor_shape_validator import assert_tensor_shape
from utill.logging_config import LoggerManager

class PathLossManager:
    """
    Advanced path loss modeling for wireless communication systems
    """
    
    def __init__(
        self, 
        system_params: Optional[SystemParameters] = None
    ):
        """
        Initialize path loss manager with system parameters
        
        Args:
            system_params (Optional[SystemParameters]): System configuration
        """
        # Initialize system parameters
        self.system_params = system_params or SystemParameters()
        
        # Initialize logger
        self.logger = LoggerManager.get_logger(__name__)
        
        # Set carrier frequency from system parameters
        self.carrier_frequency = self.system_params.carrier_frequency
        
        # Initialize panel arrays and scenarios
        self._setup_antenna_arrays()
        self._setup_scenarios()
    
    def _setup_antenna_arrays(self):
        """
        Configure antenna arrays for path loss calculations
        """
        # User Terminal (UT) Array
        self.ut_array = tr38901.PanelArray(
            num_rows=1,
            num_cols=1,
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization_type="VH",
            polarization="dual",
            antenna_pattern="38.901",
            carrier_frequency=self.system_params.carrier_frequency
        )
        
        # Base Station (BS) Array
        self.bs_array = tr38901.PanelArray(
            num_rows=1,
            num_cols=1,
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization_type="VH",
            polarization="dual",
            antenna_pattern="38.901",
            carrier_frequency=self.system_params.carrier_frequency
        )
    
    def _setup_scenarios(self):
        """
        Initialize path loss scenarios
        """
        # Urban Micro (UMi) Scenario
        self.umi_scenario = tr38901.UMiScenario(
            carrier_frequency=self.system_params.carrier_frequency,
            o2i_model="low",
            ut_array=self.ut_array,
            bs_array=self.bs_array,
            direction="downlink"
        )
        
        # Urban Macro (UMa) Scenario
        self.uma_scenario = tr38901.UMaScenario(
            carrier_frequency=self.system_params.carrier_frequency,
            o2i_model="low",
            ut_array=self.ut_array,
            bs_array=self.bs_array,
            direction="downlink"
        )
    
    def calculate_free_space_path_loss(
        self, 
        distance: tf.Tensor, 
        frequency: Optional[float] = None
    ) -> tf.Tensor:
        """
        Calculate Free Space Path Loss (FSPL) with enhanced physical constraints and validation.
        
        Args:
            distance (tf.Tensor): Distance between transmitter and receiver (meters)
            frequency (Optional[float]): Carrier frequency in Hz (defaults to system frequency)
        
        Returns:
            tf.Tensor: Free Space Path Loss (FSPL) in dB with physical constraints
        """
        try:
            # Physical constants
            c = 299792458.0  # Speed of light in m/s

            # Get frequency first
            if frequency is None:
                frequency = tf.cast(self.system_params.carrier_frequency, tf.float32)
            else:
                frequency = tf.cast(frequency, tf.float32)
                
            # Calculate wavelength and critical distance
            wavelength = c / frequency
            critical_distance = wavelength / (4.0 * np.pi)
            
            # Distance constraints
            min_distance = tf.maximum(critical_distance * 1.1, 1.0)  # Ensure above critical distance
            max_distance = 100000.0  # 100 km maximum realistic distance
            
            # Input validation and conversion
            distance = tf.cast(distance, tf.float32)
            distance = tf.maximum(distance, min_distance)
            distance = tf.minimum(distance, max_distance)
            
            # Pre-calculation validation
            valid_distances = tf.logical_and(
                tf.math.is_finite(distance),
                distance > 0
            )
            
            # FSPL calculation with enhanced stability
            # FSPL = 20log₁₀(4πd/λ) = 20log₁₀(4πdf/c)
            fspl_db = tf.where(
                valid_distances,
                20.0 * tf.math.log(4.0 * np.pi * distance * frequency / c) / tf.math.log(10.0),
                tf.fill(tf.shape(distance), tf.cast(60.0, tf.float32))  # Default value for invalid distances
            )
            
            # Post-calculation constraints based on physics
            min_fspl = 20.0  # Minimum realistic path loss in dB
            max_fspl = 160.0  # Maximum realistic path loss in dB
            fspl_db = tf.clip_by_value(fspl_db, min_fspl, max_fspl)
            
            # Add validation check
            zero_values = tf.reduce_sum(tf.cast(tf.equal(fspl_db, 0.0), tf.int32))
            if zero_values > 0:
                self.logger.warning(f"Found {zero_values} zero values in FSPL calculation") 
            
            # Add shape validation
            if not tf.reduce_all(tf.math.is_finite(fspl_db)):
                self.logger.error("Non-finite values detected in FSPL calculation")
                fspl_db = tf.where(tf.math.is_finite(fspl_db), fspl_db, min_fspl)
            
            # Add frequency-dependent adjustment
            freq_ghz = frequency / 1e9
            freq_adjustment = 20.0 * tf.math.log(freq_ghz) / tf.math.log(10.0)
            fspl_db += freq_adjustment
            
            # Final validation and cleanup
            fspl_db = tf.where(tf.math.is_finite(fspl_db), fspl_db, min_fspl)
            
            # Log calculation details for debugging
            self.logger.debug(
                f"FSPL calculation details:\n"
                f"Critical distance: {critical_distance:.2e} m\n"
                f"Min distance used: {min_distance:.2f} m\n"
                f"Distance range: {tf.reduce_min(distance).numpy():.2f} - "
                f"{tf.reduce_max(distance).numpy():.2f} m\n"
                f"FSPL range: {tf.reduce_min(fspl_db).numpy():.2f} - "
                f"{tf.reduce_max(fspl_db).numpy():.2f} dB"
            )
            
            # Enhanced logging right after calculation
            debug_info = {
                'pre_clip_min': tf.reduce_min(fspl_db).numpy(),
                'pre_clip_max': tf.reduce_max(fspl_db).numpy(),
                'zero_count_pre': tf.reduce_sum(tf.cast(tf.equal(fspl_db, 0.0), tf.int32)).numpy(),
                'input_distance_min': tf.reduce_min(distance).numpy(),
                'input_distance_max': tf.reduce_max(distance).numpy(),
            }
            
            self.logger.debug(
                f"\nFSPL Calculation Debug:"
                f"\n - Input distances range: [{debug_info['input_distance_min']:.2f}, {debug_info['input_distance_max']:.2f}] m"
                f"\n - Pre-clip FSPL range: [{debug_info['pre_clip_min']:.2f}, {debug_info['pre_clip_max']:.2f}] dB"
                f"\n - Zero values before clip: {debug_info['zero_count_pre']}"
            )
            
            # After final clipping
            final_zero_count = tf.reduce_sum(tf.cast(tf.equal(fspl_db, 0.0), tf.int32)).numpy()
            if final_zero_count > 0:
                self.logger.warning(f"Found {final_zero_count} zero values in FSPL after all processing")
                
            return fspl_db
            
        except Exception as e:
            self.logger.error(f"FSPL calculation failed: {str(e)}")
            raise

    def calculate_path_loss(self, distance: tf.Tensor, scenario: str = 'umi') -> tf.Tensor:
        """
        Calculate path loss for different scenarios with improved shape handling
        
        Args:
            distance (tf.Tensor): Distance between transmitter and receiver in meters
            scenario (str): Path loss scenario ('umi' or 'uma')
        
        Returns:
            tf.Tensor: Path loss in dB with shape [batch_size]
        """
        try:
            # Input validation and reshaping
            distance = tf.cast(distance, dtype=tf.float32)
            distance = tf.reshape(distance, [-1])
            batch_size = tf.shape(distance)[0]
            
            # Apply minimum distance constraint
            min_distance = 1.0
            distance = tf.maximum(distance, min_distance)
            
            # Set heights based on scenario
            bs_height = 10.0 if scenario.lower() == 'umi' else 25.0
            ut_height = 1.5
            
            # Create location tensors
            ut_locations = tf.stack([
                distance,
                tf.zeros_like(distance),
                tf.ones_like(distance) * ut_height
            ], axis=1)
            ut_locations = tf.expand_dims(ut_locations, axis=0)
            
            bs_locations = tf.tile(
                tf.constant([[[0., 0., bs_height]]], dtype=tf.float32),
                [1, batch_size, 1]
            )
            
            # Select appropriate scenario
            scenario_obj = self.umi_scenario if scenario.lower() == 'umi' else self.uma_scenario
            
            # Create auxiliary tensors
            zero_orientations = tf.zeros([1, batch_size, 3], dtype=tf.float32)
            zero_velocities = tf.zeros([1, batch_size, 3], dtype=tf.float32)
            indoor_state = tf.zeros([1, batch_size], dtype=tf.bool)
            
            # Set topology
            scenario_obj.set_topology(
                ut_loc=ut_locations,
                bs_loc=bs_locations,
                in_state=indoor_state,
                ut_orientations=zero_orientations,
                bs_orientations=zero_orientations,
                ut_velocities=zero_velocities
            )
            
            # Calculate basic path loss
            path_loss = tf.squeeze(scenario_obj.basic_pathloss)
            
            # Apply frequency correction
            freq_ghz = self.carrier_frequency / 1e9
            freq_correction = 20.0 * tf.math.log(freq_ghz) / tf.math.log(10.0)
            
            # Generate shadow fading
            shadow_std = 4.0 if scenario.lower() == 'umi' else 6.0
            shadow_fading = tf.random.normal(
                [batch_size],
                mean=0.0,
                stddev=shadow_std,
                dtype=tf.float32
            )
            
            # Combine all components
            total_path_loss = path_loss + freq_correction + shadow_fading
            
            # Clip values to reasonable range
            total_path_loss = tf.clip_by_value(total_path_loss, 20.0, 160.0)
            
            # Add validation
            if not tf.reduce_all(tf.math.is_finite(total_path_loss)):
                self.logger.error("Non-finite values detected in path loss calculation")
                total_path_loss = tf.where(
                    tf.math.is_finite(total_path_loss),
                    total_path_loss,
                    60.0  # Default value for invalid calculations
                )

            return tf.reshape(total_path_loss, [-1])
            
        except Exception as e:
            self.logger.error(f"Path loss calculation failed: {str(e)}")
            raise
    
    def apply_path_loss(
        self, 
        channel_response: tf.Tensor, 
        distance: tf.Tensor, 
        scenario: str = 'umi'
    ) -> tf.Tensor:
        """
        Apply path loss to channel response with proper shape handling
        
        Args:
            channel_response (tf.Tensor): MIMO channel matrix [batch_size, num_rx, num_tx]
            distance (tf.Tensor): Distance between TX and RX [batch_size]
            scenario (str): Path loss scenario ('umi' or 'uma')
        
        Returns:
            tf.Tensor: Channel response with applied path loss
        """
        try:
            # Get batch size from channel response
            batch_size = tf.shape(channel_response)[0]
            
            # Log input shapes
            self.logger.debug(f"Initial channel_response shape: {channel_response.shape}")
            self.logger.debug(f"Initial distance shape: {distance.shape}")
            
            # Ensure distance tensor has correct shape
            distance = tf.reshape(distance[:batch_size], [-1])
            
            # Calculate path loss for the batch
            path_loss_db = self.calculate_path_loss(distance, scenario)
            
            # Convert to linear scale
            path_loss_linear = tf.pow(10.0, -path_loss_db / 20.0)
            
            # Explicitly reshape path loss to match batch size
            path_loss_linear = tf.reshape(path_loss_linear, [-1])[:batch_size]
            
            # Add dimensions for broadcasting
            path_loss_shaped = tf.expand_dims(tf.expand_dims(path_loss_linear, axis=1), axis=2)
            
            # Apply path loss to channel response
            attenuated_channel = channel_response * tf.cast(
                path_loss_shaped,
                dtype=channel_response.dtype
            )
            
            return attenuated_channel
            
        except Exception as e:
            self.logger.error(f"Error applying path loss: {str(e)}")
            self.logger.error(f"Shapes - channel_response: {channel_response.shape}, "
                            f"distance: {distance.shape}")
            raise
        
    def generate_path_loss_statistics(
        self, 
        min_distance: float = 10.0, 
        max_distance: float = 500.0, 
        num_samples: int = 1000
    ) -> dict:
        """
        Generate comprehensive path loss statistics with improved sampling
        
        Args:
            min_distance (float): Minimum distance in meters
            max_distance (float): Maximum distance in meters
            num_samples (int): Number of samples
        
        Returns:
            dict: Path loss statistics
        """
        try:
            # Generate logarithmically spaced distances for better sampling
            distances = tf.exp(tf.linspace(
                tf.math.log(float(min_distance)),
                tf.math.log(float(max_distance)),
                num_samples
            ))
            
            # Calculate path loss for different scenarios
            umi_path_loss = self.calculate_path_loss(distances, 'umi')
            uma_path_loss = self.calculate_path_loss(distances, 'uma')
            fspl = self.calculate_free_space_path_loss(distances)
            
            # Calculate combined path losses
            combined_umi = 0.7 * fspl + 0.3 * umi_path_loss
            combined_uma = 0.7 * fspl + 0.3 * uma_path_loss
            
            # Generate comprehensive statistics
            statistics = {
                'distances': distances.numpy(),
                'umi_path_loss': umi_path_loss.numpy(),
                'uma_path_loss': uma_path_loss.numpy(),
                'fspl': fspl.numpy(),
                'combined_umi': combined_umi.numpy(),
                'combined_uma': combined_uma.numpy(),
                'statistics': {
                    'umi_mean': tf.reduce_mean(umi_path_loss).numpy(),
                    'umi_std': tf.math.reduce_std(umi_path_loss).numpy(),
                    'uma_mean': tf.reduce_mean(uma_path_loss).numpy(),
                    'uma_std': tf.math.reduce_std(uma_path_loss).numpy(),
                    'fspl_mean': tf.reduce_mean(fspl).numpy(),
                    'fspl_std': tf.math.reduce_std(fspl).numpy()
                }
            }
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Error generating path loss statistics: {str(e)}")
            raise        
# Example usage
def main():
    # Create path loss manager with default parameters
    path_loss_manager = PathLossManager()
    
    # Generate path loss statistics
    stats = path_loss_manager.generate_path_loss_statistics()
    
    # Print basic statistics
    for key, value in stats.items():
        print(f"{key}:")
        print(f"  Mean: {np.mean(value)}")
        print(f"  Std: {np.std(value)}")
        print(f"  Min: {np.min(value)}")
        print(f"  Max: {np.max(value)}")

if __name__ == "__main__":
    main()