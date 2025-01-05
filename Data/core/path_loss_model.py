# Path Loss Calculations
# path_loss_model.py
# Implements path loss modeling with a focus on FSPL (Free Space Path Loss) 
# for simplicity and applicability to the project's goals.
#
# Key Features:
# - Supports FSPL calculation with physical constraints for realistic modeling.
# - Validates and clips path loss values to ensure parameters remain within expected ranges.
# - Simplified handling of optional scenarios (UMi, UMa) for flexibility in future extensions.
# - Per-antenna path loss calculations are included but optional.
# - Logging integrated for debugging and tracking parameter consistency.
#
# Constraints:
# - Path loss values are constrained between 20 dB and 160 dB.
# - Invalid path loss values are corrected or replaced with defaults.
#
# Updates:
# - Removed unused functionality related to urban macro and micro models if not required.
# - Streamlined validation and logging to align with the project's dataset generation focus.
# - Focused implementation on FSPL for clear, reproducible results.

# Path Loss Calculations
# path_loss_model.py
# Implements path loss modeling with a focus on FSPL (Free Space Path Loss).

import tensorflow as tf
import numpy as np
from typing import Optional
from config.system_parameters import SystemParameters
from utill.logging_config import LoggerManager


class PathLossManager:
    """
    Path Loss Manager for Free Space Path Loss (FSPL) calculations.
    """

    def __init__(self, system_params: Optional[SystemParameters] = None):
        """
        Initialize path loss manager with system parameters.

        Args:
            system_params (Optional[SystemParameters]): System configuration.
        """
        self.system_params = system_params or SystemParameters()
        self.logger = LoggerManager.get_logger(__name__)
        self.carrier_frequency = self.system_params.carrier_frequency

    def calculate_free_space_path_loss(self, distance: tf.Tensor) -> tf.Tensor:
        """
        Calculate Free Space Path Loss (FSPL).

        Args:
            distance (tf.Tensor): Distance between transmitter and receiver (meters).

        Returns:
            tf.Tensor: FSPL in dB.
        """
        try:
            c = 299792458.0  # Speed of light (m/s)
            frequency = tf.cast(self.carrier_frequency, tf.float32)
            wavelength = c / frequency

            # Ensure distance is within valid range
            min_distance = wavelength / (4.0 * np.pi)
            distance = tf.clip_by_value(distance, min_distance, 100000.0)  # 100 km max

            # FSPL formula
            fspl_db = 20.0 * tf.math.log(4.0 * np.pi * distance * frequency / c) / tf.math.log(10.0)
            fspl_db = tf.clip_by_value(fspl_db, 20.0, 160.0)  # Constrain FSPL

            return fspl_db

        except Exception as e:
            self.logger.error(f"FSPL calculation failed: {str(e)}")
            raise

# Example usage
def main():
    # Create PathLossManager
    system_params = SystemParameters()
    path_loss_manager = PathLossManager(system_params)

    # Example distances (in meters)
    distances = tf.constant([1.0, 10.0, 100.0, 1000.0, 10000.0], dtype=tf.float32)
    fspl = path_loss_manager.calculate_free_space_path_loss(distances)

    # Print results
    print("Distances (m):", distances.numpy())
    print("FSPL (dB):", fspl.numpy())


if __name__ == "__main__":
    main()
