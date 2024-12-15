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
        Calculate Free Space Path Loss (FSPL)
        
        Args:
            distance (tf.Tensor): Distance between transmitter and receiver
            frequency (Optional[float]): Carrier frequency (defaults to system frequency)
        
        Returns:
            tf.Tensor: Free Space Path Loss in dB
        """
        # Use system carrier frequency if not specified
        frequency = frequency or self.system_params.carrier_frequency
        
        # Speed of light
        c = 3e8
        
        # Wavelength calculation
        wavelength = c / frequency
        
        # Avoid division by zero
        distance = tf.maximum(distance, 1e-3)
        
        # FSPL calculation
        fspl_db = 20.0 * tf.math.log(distance / wavelength) / tf.math.log(10.0)
        
        return fspl_db
    
    def calculate_path_loss(self, distance: tf.Tensor, scenario: str = 'umi') -> tf.Tensor:
        """
        Calculate path loss for different scenarios with improved handling
        
        Args:
            distance (tf.Tensor): Distance between transmitter and receiver in meters
            scenario (str): Path loss scenario ('umi' or 'uma')
        
        Returns:
            tf.Tensor: Path loss in dB
        """
        # Validate input tensor
        assert_tensor_shape(distance, [None], 'distance')
        
        # Ensure minimum distance and reshape
        min_distance = 1.0  # Minimum 1 meter
        distance = tf.maximum(distance, min_distance)
        
        # Convert distance to 3D coordinates for multiple samples
        batch_size = tf.shape(distance)[0]
        
        # Define heights based on scenario
        if scenario.lower() == 'umi':
            bs_height = 10.0  # Urban Micro BS height (typically 10m)
            ut_height = 1.5   # User Terminal height (typically 1.5m)
        else:  # UMa scenario
            bs_height = 25.0  # Urban Macro BS height (typically 25m)
            ut_height = 1.5   # User Terminal height (typically 1.5m)
        
        # Create location tensors
        ut_locations = tf.stack([
            distance,
            tf.zeros_like(distance),
            tf.ones_like(distance) * ut_height
        ], axis=1)
        ut_locations = tf.expand_dims(ut_locations, axis=0)  # Add batch dimension
        
        bs_locations = tf.tile(
            tf.constant([[[0., 0., bs_height]]]), 
            [1, batch_size, 1]
        )
        
        # Set topology for path loss calculation
        scenario_obj = (
            self.umi_scenario if scenario.lower() == 'umi' 
            else self.uma_scenario
        )
        
        # Create orientation and velocity tensors
        zero_orientations = tf.zeros([1, batch_size, 3])
        zero_velocities = tf.zeros([1, batch_size, 3])
        indoor_state = tf.zeros([1, batch_size], dtype=tf.bool)  # Outdoor by default
        
        try:
            scenario_obj.set_topology(
                ut_loc=ut_locations,
                bs_loc=bs_locations,
                in_state=indoor_state,
                ut_orientations=zero_orientations,
                bs_orientations=zero_orientations,
                ut_velocities=zero_velocities
            )
            
            # Get basic path loss
            path_loss = tf.squeeze(scenario_obj.basic_pathloss)
            
            # Apply frequency-dependent correction
            freq_ghz = self.carrier_frequency / 1e9
            freq_correction = 20 * tf.math.log(freq_ghz) / tf.math.log(10.0)
            
            # Apply additional losses
            shadow_fading = tf.random.normal(
                tf.shape(path_loss),
                mean=0.0,
                stddev=4.0 if scenario.lower() == 'umi' else 6.0
            )
            
            # Combine all losses
            total_path_loss = path_loss + freq_correction + shadow_fading
            
            # Ensure path loss is within realistic bounds
            min_path_loss = 20.0  # Minimum path loss in dB
            max_path_loss = 160.0  # Maximum path loss in dB
            total_path_loss = tf.clip_by_value(total_path_loss, min_path_loss, max_path_loss)
            
            return total_path_loss
            
        except Exception as e:
            self.logger.error(f"Path loss calculation failed: {str(e)}")
            self.logger.error("Detailed error traceback:", exc_info=True)
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
            
            # Calculate path loss in dB using the first batch_size distances
            path_loss_db = self.calculate_path_loss(distance[:batch_size], scenario)
            
            # Convert path loss from dB to linear scale
            path_loss_linear = tf.pow(10.0, -path_loss_db / 20.0)
            
            # Reshape path loss for broadcasting using gather instead of slice
            path_loss_linear = tf.gather(path_loss_linear, tf.range(batch_size))
            path_loss_linear = tf.expand_dims(tf.expand_dims(path_loss_linear, -1), -1)
            
            # Apply path loss to channel response
            attenuated_channel = channel_response * tf.cast(
                path_loss_linear, 
                dtype=channel_response.dtype
            )
            
            return attenuated_channel
            
        except Exception as e:
            self.logger.error(f"Error applying path loss: {str(e)}")
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