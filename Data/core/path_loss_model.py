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
        self.system_params = system_params or SystemParameters()
        
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
    
    def calculate_path_loss(
        self, 
        distance: tf.Tensor, 
        scenario: str = 'umi'
    ) -> tf.Tensor:
        """
        Calculate path loss for different scenarios
        
        Args:
            distance (tf.Tensor): Distance between transmitter and receiver
            scenario (str): Path loss scenario ('umi' or 'uma')
        
        Returns:
            tf.Tensor: Path loss in dB
        """
        # Validate input tensor
        assert_tensor_shape(distance, [None], 'distance')
        
        # Set topology for path loss calculation
        scenario_obj = (
            self.umi_scenario if scenario.lower() == 'umi' 
            else self.uma_scenario
        )
        
        scenario_obj.set_topology(
            ut_loc=tf.constant([[[distance.numpy()[0], 0., 1.5]]]),
            bs_loc=tf.constant([[[0., 0., 25.]]]),
            in_state=tf.constant([[False]]),
            ut_orientations=tf.constant([[[0., 0., 0.]]]),
            bs_orientations=tf.constant([[[0., 0., 0.]]]),
            ut_velocities=tf.constant([[[0., 0., 0.]]])
        )
        
        # Get basic path loss
        basic_path_loss = scenario_obj.basic_pathloss
        
        return tf.squeeze(basic_path_loss)
    def apply_path_loss(
        self, 
        channel_response: tf.Tensor, 
        distance: tf.Tensor, 
        scenario: str = 'umi'
    ) -> tf.Tensor:
        """
        Apply path loss to channel response
        
        Args:
            channel_response (tf.Tensor): MIMO channel matrix
            distance (tf.Tensor): Distance between transmitter and receiver
            scenario (str): Path loss scenario
        
        Returns:
            tf.Tensor: Channel response with applied path loss
        """
        # Calculate Free Space and Scenario Path Loss
        fspl_db = self.calculate_free_space_path_loss(distance)
        scenario_pl_db = self.calculate_path_loss(distance, scenario)
        
        # Combined path loss
        total_path_loss_db = fspl_db + scenario_pl_db
        
        # Convert to linear scale
        path_loss_linear = tf.pow(10.0, -total_path_loss_db / 20.0)
        
        # Reshape for broadcasting
        path_loss_linear = tf.reshape(path_loss_linear, [-1, 1, 1])
        
        # Apply path loss to channel response
        return channel_response * tf.cast(path_loss_linear, dtype=channel_response.dtype)
    
    def generate_path_loss_statistics(
        self, 
        min_distance: float = 10.0, 
        max_distance: float = 500.0, 
        num_samples: int = 1000
    ) -> dict:
        """
        Generate comprehensive path loss statistics
        
        Args:
            min_distance (float): Minimum distance
            max_distance (float): Maximum distance
            num_samples (int): Number of samples
        
        Returns:
            dict: Path loss statistics
        """
        distances = tf.linspace(min_distance, max_distance, num_samples)
        
        # Calculate path loss for different scenarios
        umi_path_loss = self.calculate_path_loss(distances, 'umi')
        uma_path_loss = self.calculate_path_loss(distances, 'uma')
        fspl = self.calculate_free_space_path_loss(distances)
        
        return {
            'distances': distances.numpy(),
            'umi_path_loss': umi_path_loss.numpy(),
            'uma_path_loss': uma_path_loss.numpy(),
            'fspl': fspl.numpy()
        }

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