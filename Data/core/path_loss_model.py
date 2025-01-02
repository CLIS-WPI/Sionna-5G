# Path loss calculations
# core/path_loss_model.py
# Advanced Path Loss Modeling and Calculation
# Implements multiple path loss scenarios with comprehensive statistical analysis
# Provides flexible path loss computation for different wireless communication environments,
# including per-antenna path loss calculations for detailed analysis.
#
# Key Features:
# - Supports urban microcell (UMi) and urban macrocell (UMa) scenarios.
# - Calculates path loss values per antenna pair (receiver-transmitter).
# - Includes distance validation and clipping to ensure realistic parameters.
# - Incorporates shadow fading and frequency correction for realistic modeling.
# - Provides detailed logging and validation for each step of the calculation.
#
# Path Loss Constraints:
# - Path loss values must be between 20 dB and 160 dB.
# - Zero or invalid path loss indicates an error and must be corrected.
#
# Improvements:
# - Added per-antenna path loss calculations for more granular modeling.
# - Integrated frequency correction and shadow fading as customizable components.
# - Enhanced validation thresholds to identify and log invalid path loss values.
# - Comprehensive logging for debugging and performance tracking.


import tensorflow as tf
import numpy as np
from sionna.channel import tr38901
from typing import Optional, Tuple, Callable
from config.system_parameters import SystemParameters
from utill.tensor_shape_validator import assert_tensor_shape
from utill.logging_config import LoggerManager
import traceback

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
        self.max_batch_size = 64000
        self.min_batch_size = 16000
        
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

    def calculate_path_loss(
        self,
        distance: tf.Tensor,
        scenario: str = 'umi',
        path_loss_params: Optional[dict] = None
    ) -> tf.Tensor:
        """
        Calculate path loss for different scenarios with per-antenna granularity.
        """
        try:
            print(f"\nDEBUG - Entering calculate_path_loss:")
            print(f"Distance tensor shape: {tf.shape(distance)}")
            print(f"Scenario: {scenario}")

            # Ensure proper tensor shape and type
            if not isinstance(distance, tf.Tensor):
                distance = tf.convert_to_tensor(distance, dtype=tf.float32)

            batch_size = tf.shape(distance)[0]

            # Create location tensors
            ut_locations = tf.stack([
                distance,
                tf.zeros_like(distance),
                tf.ones_like(distance) * 1.5  # UT height 1.5m
            ], axis=1)
            
            bs_locations = tf.zeros([batch_size, 3])
            bs_height = 10.0 if scenario == 'umi' else 25.0
            bs_locations = tf.tensor_scatter_nd_update(
                bs_locations,
                [[i, 2] for i in range(batch_size)],
                tf.ones([batch_size]) * bs_height
            )

            # Select scenario
            scenario_obj = self.umi_scenario if scenario == 'umi' else self.uma_scenario

            # Set topology
            scenario_obj.set_topology(
                ut_loc=tf.expand_dims(ut_locations, axis=0),
                bs_loc=tf.expand_dims(bs_locations, axis=0),
                ut_orientations=tf.zeros([1, batch_size, 3]),
                bs_orientations=tf.zeros([1, batch_size, 3]),
                ut_velocities=tf.zeros([1, batch_size, 3]),
                in_state=tf.zeros([1, batch_size], dtype=tf.bool)
            )

            # Calculate path loss using the correct method
            path_loss = scenario_obj.pathloss  # Changed to use the pathloss attribute
            path_loss = tf.squeeze(path_loss)  # Remove unnecessary dimensions

            # Add frequency correction
            freq_ghz = self.carrier_frequency / 1e9
            freq_correction = 20.0 * tf.math.log(freq_ghz) / tf.math.log(10.0)
            path_loss += freq_correction

            # Add shadow fading
            shadow_std = 4.0 if scenario == 'umi' else 6.0
            shadow_fading = tf.random.normal(
                tf.shape(path_loss), 
                mean=0.0, 
                stddev=shadow_std, 
                dtype=tf.float32
            )
            path_loss += shadow_fading

            # Clip values
            path_loss = tf.clip_by_value(path_loss, 20.0, 160.0)

            # Reshape for antenna dimensions if needed
            if len(path_loss.shape) == 1:
                path_loss = tf.expand_dims(tf.expand_dims(path_loss, axis=1), axis=2)
                path_loss = tf.broadcast_to(
                    path_loss, 
                    [batch_size, self.system_params.num_rx, self.system_params.num_tx]
                )

            return path_loss

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

    # In path_loss_model.py, enhance validation:
    def _validate_path_loss_values(self, path_loss_db: tf.Tensor, source: str) -> Tuple[bool, tf.Tensor]:
        """
        Validate path loss values with enhanced physical constraints and error reporting
        
        Args:
            path_loss_db: Path loss values in dB
            source: Source identifier for logging
            
        Returns:
            Tuple containing validation status and corrected tensor
        """
        try:
            # Physical constraints
            MIN_PATH_LOSS = 20.0  # Minimum realistic path loss in dB
            MAX_PATH_LOSS = 160.0  # Maximum realistic path loss in dB
            
            # Initial shape validation
            if not isinstance(path_loss_db, tf.Tensor):
                path_loss_db = tf.convert_to_tensor(path_loss_db, dtype=tf.float32)
                
            original_shape = tf.shape(path_loss_db)
            path_loss_db = tf.reshape(path_loss_db, [-1])  # Flatten for processing
            
            # Check for invalid values
            is_finite = tf.math.is_finite(path_loss_db)
            non_finite_count = tf.reduce_sum(tf.cast(~is_finite, tf.int32))
            
            if non_finite_count > 0:
                self.logger.warning(f"{source}: Found {non_finite_count} non-finite values")
                # Replace non-finite values with MIN_PATH_LOSS
                path_loss_db = tf.where(is_finite, path_loss_db, MIN_PATH_LOSS)
            
            # Check for zeros
            zero_mask = tf.equal(path_loss_db, 0.0)
            zero_count = tf.reduce_sum(tf.cast(zero_mask, tf.int32))
            
            if zero_count > 0:
                self.logger.warning(f"{source}: Found {zero_count} zero values")
                # Replace zeros with MIN_PATH_LOSS
                path_loss_db = tf.where(zero_mask, MIN_PATH_LOSS, path_loss_db)
            
            # Clip values to physical bounds
            original_path_loss = path_loss_db
            path_loss_db = tf.clip_by_value(path_loss_db, MIN_PATH_LOSS, MAX_PATH_LOSS)
            
            # Count clipped values
            clipped_low = tf.reduce_sum(tf.cast(original_path_loss < MIN_PATH_LOSS, tf.int32))
            clipped_high = tf.reduce_sum(tf.cast(original_path_loss > MAX_PATH_LOSS, tf.int32))
            
            if clipped_low > 0 or clipped_high > 0:
                self.logger.warning(
                    f"{source} path loss clipping statistics:\n"
                    f" - Values below {MIN_PATH_LOSS} dB: {clipped_low}\n"
                    f" - Values above {MAX_PATH_LOSS} dB: {clipped_high}"
                )
            
            # Reshape back to original shape
            path_loss_db = tf.reshape(path_loss_db, original_shape)
            
            # Final validation
            final_validation = tf.logical_and(
                tf.math.is_finite(path_loss_db),
                tf.logical_and(
                    tf.greater_equal(path_loss_db, MIN_PATH_LOSS),
                    tf.less_equal(path_loss_db, MAX_PATH_LOSS)
                )
            )
            
            is_valid = tf.reduce_all(final_validation)
            
            return is_valid, path_loss_db
            
        except Exception as e:
            self.logger.error(f"Path loss validation error: {str(e)}")
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