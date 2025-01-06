# System Parameters Configuration
# system_parameters.py
# Defines system-level parameters for MIMO communication and dataset generation.
#
# Key Features:
# - Configurable MIMO settings, including antenna configuration, OFDM parameters, and channel characteristics.
# - Supports reproducible dataset generation with global random seed management.
# - Comprehensive validation for all parameters to ensure physical plausibility and simulation consistency.
# - Includes utility methods for dynamic parameter updates and generating channel-specific configurations.
#
# Simplifications:
# - Removed hardware-related parameters and memory scaling for streamlined implementation.
# - Focused solely on simulation and dataset generation parameters based on the project goals.
#
# Updates:
# - Enhanced parameter validation to align with the simulation plan (e.g., SNR range, modulation schemes).
# - Added flexibility for updating and retrieving configuration dictionaries for external integration.
#
# Scope:
# - This file provides the foundational configuration for system-level parameters in the MIMO simulation pipeline.
# - Advanced hardware tuning and runtime parameter updates are excluded for simplicity.

import dataclasses
from typing import Tuple, List, Optional, Dict
from datetime import datetime
import numpy as np
import tensorflow as tf
import logging


@dataclasses.dataclass
class SystemParameters:
    """
    Configuration for MIMO system and dataset generation.
    """

    # Dataset Generation Parameters
    total_samples: int = 1_320_000           # Total number of samples to generate
    batch_size: int = 1000                   # Processing batch size
    samples_per_modulation: int = None       # Samples per modulation scheme

    # Antenna Configuration
    num_tx_antennas: int = 4                 # Number of transmit antennas
    num_rx_antennas: int = 4                          # Number of receive antennas
    num_streams: int = 4                     # Number of data streams
    element_spacing: float = 0.5             # Antenna element spacing (wavelengths)
    channel_model: str = "rayleigh"          # Channel model type
    # Frequency Parameters
    carrier_frequency: float = 3.5e9         # Carrier frequency in Hz
    polarization: str = "single"             # Antenna polarization type
    # OFDM Parameters
    num_subcarriers: int = 64                # Number of subcarriers
    num_ofdm_symbols: int = 14               # OFDM symbols per slot
    subcarrier_spacing: float = 30e3         # Subcarrier spacing in Hz

    # Channel Parameters
    num_paths: int = 10                      # Number of multipath components
    snr_range: Tuple[float, float] = (0.0, 30.0)  # Signal-to-Noise Ratio range in dB
    noise_floor: float = -174                # Noise floor in dBm/Hz

    # Modulation Schemes
    modulation_schemes: List[str] = dataclasses.field(
        default_factory=lambda: ["QPSK", "16QAM", "64QAM"]
    )

    # Reproducibility
    random_seed: int = 42                    # Random seed for reproducibility

    def __post_init__(self):
        """
        Post-initialization validation and calculations.
        """
        # Automatically calculate samples per modulation if not specified
        if self.samples_per_modulation is None:
            self.samples_per_modulation = self.total_samples // len(self.modulation_schemes)

        # Validate parameters
        self._validate_parameters()

        # Set global random seeds
        self.set_global_seeds()

    def _validate_parameters(self):
        """
        Validate system parameters for consistency and physical plausibility.
        """
        # Validate antenna configuration
        assert self.num_tx > 0, "Number of transmit antennas must be positive."
        assert self.num_rx > 0, "Number of receive antennas must be positive."
        assert 1 <= self.num_streams <= min(self.num_tx, self.num_rx), "Invalid number of streams."

        # Validate frequency
        assert 1e9 <= self.carrier_frequency <= 6e9, "Carrier frequency must be in the 1-6 GHz range."

        # Validate OFDM configuration
        assert self.num_subcarriers > 0, "Number of subcarriers must be positive."
        assert self.num_ofdm_symbols > 0, "Number of OFDM symbols must be positive."
        assert self.subcarrier_spacing > 0, "Subcarrier spacing must be positive."

        # Validate SNR range
        assert -20 <= self.snr_range[0] < self.snr_range[1] <= 40, "Unrealistic SNR range."

        # Validate element spacing
        assert 0.1 <= self.element_spacing <= 1.0, "Element spacing must be between 0.1 and 1.0 wavelengths."

        # Validate modulation schemes
        valid_schemes = {"BPSK", "QPSK", "16QAM", "64QAM", "256QAM"}
        for scheme in self.modulation_schemes:
            if scheme.upper() not in valid_schemes:
                raise ValueError(f"Invalid modulation scheme: {scheme}")

    def set_global_seeds(self):
        """
        Set global random seeds for reproducibility.
        """
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        logging.info(f"Global random seed set to: {self.random_seed}")

    def get_config_dict(self) -> Dict[str, any]:
        """
        Generate a dictionary of system parameters for serialization.

        Returns:
            Dict[str, any]: Serializable system configuration.
        """
        return dataclasses.asdict(self)

    def update(self, **kwargs):
        """
        Dynamically update system parameters.

        Args:
            kwargs: Key-value pairs of parameters to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logging.warning(f"Ignoring unknown parameter: {key}")

        # Re-validate parameters after update
        self._validate_parameters()

        logging.info("System parameters updated successfully.")

    def get_channel_generation_config(self) -> Dict[str, any]:
        """
        Generate a configuration dictionary for channel generation.

        Returns:
            Dict[str, any]: Detailed configuration for channel generation.
        """
        return {
            "antenna_config": {
                "num_tx": self.num_tx,
                "num_rx": self.num_rx,
                "num_streams": self.num_streams,
                "element_spacing": self.element_spacing,
            },
            "channel_parameters": {
                "carrier_frequency": self.carrier_frequency,
                "num_paths": self.num_paths,
                "snr_range": self.snr_range,
                "noise_floor": self.noise_floor,
            },
            "resource_grid": {
                "num_subcarriers": self.num_subcarriers,
                "num_ofdm_symbols": self.num_ofdm_symbols,
                "subcarrier_spacing": self.subcarrier_spacing,
            },
            "modulation": {
                "schemes": self.modulation_schemes,
            },
        }

