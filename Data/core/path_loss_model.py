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
        Calculate path loss for different scenarios with enhanced validation, retry mechanisms, and detailed logging.
        
        Args:
            distance (tf.Tensor): Distance between transmitter and receiver in meters.
            scenario (str): Path loss scenario ('umi' or 'uma').
            path_loss_params (Optional[dict]): Optional override for default parameters.
            
        Returns:
            tf.Tensor: Path loss in dB with shape [batch_size].
        """
        try:
            # Load default or override parameters
            params = {
                'min_distance': 1.0,
                'max_distance': 5000.0,
                'shadow_std_umi': 4.0,
                'shadow_std_uma': 6.0,
                'min_path_loss': 20.0,
                'max_path_loss': 160.0,
                'max_batch_size': 10000,  # New: Maximum batch size
                'freq_scaling': 1.0,  # New: Frequency correction scaling
                'validation_thresholds': {  # New: Thresholds for validation
                    'min_valid_path_loss': 20.0,
                    'max_valid_path_loss': 160.0
                }
            }
            if path_loss_params:
                params.update(path_loss_params)

            # Validate scenario initialization
            if not hasattr(self, 'umi_scenario') or not hasattr(self, 'uma_scenario'):
                raise RuntimeError("Path loss scenarios are not properly initialized.")
            if not hasattr(self, 'carrier_frequency') or self.carrier_frequency <= 0:
                raise ValueError("Invalid or uninitialized carrier frequency.")

            # Persistent RNG for shadow fading
            if not hasattr(self, '_rng'):
                self._rng = tf.random.Generator.from_seed(42)

            # Input validation
            if not isinstance(distance, tf.Tensor):
                self.logger.warning(f"Converting distance input from {type(distance)} to TensorFlow tensor.")
                distance = tf.convert_to_tensor(distance, dtype=tf.float32)

            # Handle and reshape distances
            original_shape = distance.shape
            try:
                distance = tf.cast(distance, dtype=tf.float32)
                distance = tf.reshape(distance, [-1])
                batch_size = tf.shape(distance)[0]
                
                # batch size validation and adjustment
                # Enforce maximum batch size
                if batch_size > params['max_batch_size']:
                    self.logger.warning(
                        f"Batch size {batch_size} exceeds max allowed size {params['max_batch_size']}. "
                        f"Adjusting dynamically to {params['max_batch_size']}."
                    )
                    distance = distance[:params['max_batch_size']]
                    batch_size = params['max_batch_size']
                
            except Exception as e:
                raise ValueError(f"Failed to process distance shape {original_shape}: {str(e)}")

            # Distance range validation and clipping
            original_distance = distance
            distance = tf.clip_by_value(distance, params['min_distance'], params['max_distance'])

            # Log clipped distances
            clipped_mask = tf.not_equal(original_distance, distance)
            num_clipped = tf.reduce_sum(tf.cast(clipped_mask, tf.int32))
            if num_clipped > 0:
                self.logger.warning(f"Total clipped distances: {num_clipped.numpy()}")
                clipped_indices = tf.where(clipped_mask)[:5]
                clipped_orig = tf.gather(original_distance, clipped_indices)
                clipped_new = tf.gather(distance, clipped_indices)
                self.logger.warning(
                    f"Clipped distances (first 5 examples):\n" +
                    "\n".join([f"Index {idx.numpy()[0]}: {orig.numpy():.2f}m -> {new.numpy():.2f}m" 
                            for idx, orig, new in zip(clipped_indices, clipped_orig, clipped_new)])
                )

            # Scenario validation with fallback
            scenario = scenario.lower()
            if scenario not in ['umi', 'uma']:
                self.logger.warning(f"Invalid scenario '{scenario}', defaulting to 'umi'.")
                scenario = 'umi'

            # Scenario-specific parameters
            shadow_std = params['shadow_std_umi'] if scenario == 'umi' else params['shadow_std_uma']
            bs_height = 10.0 if scenario == 'umi' else 25.0
            ut_height = 1.5

            # Create location tensors
            ut_locations = tf.expand_dims(tf.stack([distance, tf.zeros_like(distance), tf.ones_like(distance) * ut_height], axis=1), axis=0)
            bs_locations = tf.tile(tf.constant([[[0.0, 0.0, bs_height]]], dtype=tf.float32), [1, batch_size, 1])

            # Validate tensor shapes
            expected_shape = [1, batch_size, 3]
            for tensor, name in [(ut_locations, 'UT'), (bs_locations, 'BS')]:
                if tensor.shape.as_list()[1:] != expected_shape[1:]:
                    raise ValueError(f"Invalid {name} locations shape: {tensor.shape} vs expected {expected_shape}")

            # Auxiliary tensors
            zero_orientations = tf.zeros([1, batch_size, 3], dtype=tf.float32)
            zero_velocities = tf.zeros([1, batch_size, 3], dtype=tf.float32)
            indoor_state = tf.zeros([1, batch_size], dtype=tf.bool)

            # Retry mechanism for topology setting
            scenario_obj = self.umi_scenario if scenario == 'umi' else self.uma_scenario
            max_retries = 3
            for retry in range(max_retries):
                try:
                    scenario_obj.set_topology(
                        ut_loc=ut_locations,
                        bs_loc=bs_locations,
                        in_state=indoor_state,
                        ut_orientations=zero_orientations,
                        bs_orientations=zero_orientations,
                        ut_velocities=zero_velocities
                    )
                    break
                except Exception as e:
                    if retry == max_retries - 1:
                        raise RuntimeError(f"Failed to set topology after {max_retries} attempts: {str(e)}")
                    self.logger.warning(f"Retry {retry + 1}/{max_retries} setting topology.")

            # Path loss components
            path_loss = tf.clip_by_value(scenario_obj.basic_pathloss, 0.0, 200.0)
            
            # Frequency correction
            freq_ghz = self.carrier_frequency / 1e9
            if not (0.1 <= freq_ghz <= 100.0):
                raise ValueError(f"Invalid frequency: {freq_ghz} GHz")
            freq_correction = params['freq_scaling'] * (20.0 * tf.math.log(freq_ghz) / tf.math.log(10.0))
            shadow_fading = self._rng.normal(shape=[batch_size], mean=0.0, stddev=shadow_std, dtype=tf.float32)

            # Combine components and clip
            total_path_loss = tf.clip_by_value(path_loss + freq_correction + shadow_fading, params['min_path_loss'], params['max_path_loss'])

            # Apply validation thresholds
            valid_mask = tf.logical_and(
                total_path_loss >= params['validation_thresholds']['min_valid_path_loss'],
                total_path_loss <= params['validation_thresholds']['max_valid_path_loss']
            )
            if not tf.reduce_all(valid_mask):
                invalid_count = tf.reduce_sum(tf.cast(~valid_mask, tf.int32))
                self.logger.warning(f"Detected {invalid_count.numpy()} path loss values outside validation thresholds.")

            # Statistics logging
            stats = {
                'mean': tf.reduce_mean(total_path_loss),
                'min': tf.reduce_min(total_path_loss),
                'max': tf.reduce_max(total_path_loss),
                'std': tf.math.reduce_std(total_path_loss)
            }
            self.logger.info("Path loss statistics:")
            for key, value in stats.items():
                self.logger.info(f"- {key}: {value:.2f} dB")

            # Log clipped path loss values
            clipped_mask = tf.not_equal(total_path_loss, path_loss + freq_correction + shadow_fading)
            num_clipped = tf.reduce_sum(tf.cast(clipped_mask, tf.int32))
            if num_clipped > 0:
                self.logger.warning(f"Total clipped path loss values: {num_clipped.numpy()}")

            return total_path_loss

        except Exception as e:
            self.logger.error(f"Path loss calculation failed: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
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
    def _validate_path_loss(self, path_loss_db: tf.Tensor, source: str) -> tf.Tensor:
        try:
            # Track original values
            original_stats = {
                'min': tf.reduce_min(path_loss_db),
                'max': tf.reduce_max(path_loss_db),
                'mean': tf.reduce_mean(path_loss_db)
            }
            
            # Count values outside physical bounds
            too_low = tf.reduce_sum(tf.cast(path_loss_db < 20.0, tf.int32))
            too_high = tf.reduce_sum(tf.cast(path_loss_db > 160.0, tf.int32))
            
            # Clip values
            path_loss_db = tf.clip_by_value(path_loss_db, 20.0, 160.0)
            
            # Log detailed statistics
            if too_low > 0 or too_high > 0:
                self.logger.warning(
                    f"{source} path loss clipping stats:\n"
                    f"Values < 20 dB: {too_low}\n"
                    f"Values > 160 dB: {too_high}\n"
                    f"Original range: [{original_stats['min']:.2f}, {original_stats['max']:.2f}] dB"
                )
                
            return path_loss_db
            
        except Exception as e:
            self.logger.error(f"Path loss validation failed: {str(e)}")
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