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
    """Configuration for MIMO system and dataset generation."""

    # MIMO System Configuration
    num_tx_antennas: int = 4                 # Number of transmit antennas
    num_rx_antennas: int = 4                 # Number of receive antennas
    num_streams: int = 4                     # Number of data streams
    element_spacing: float = 0.5             # Antenna element spacing (wavelengths)
    polarization: str = "single"             # Antenna polarization type

    # RF Parameters
    carrier_frequency: float = 3.5e9         # Carrier frequency in Hz
    subcarrier_spacing: float = 30e3         # Subcarrier spacing in Hz
    noise_floor: float = -174                # Noise floor in dBm/Hz

    # OFDM Configuration
    num_subcarriers: int = 64                # Number of subcarriers
    num_ofdm_symbols: int = 14               # OFDM symbols per slot

    # Channel Model Parameters
    channel_model: str = "rayleigh"          # Channel model type
    num_paths: int = 10                      # Number of multipath components
    snr_ranges: Dict[str, Tuple[float, float]] = dataclasses.field(
    default_factory=lambda: {
        "QPSK": (15.0, 20.0),    # Conservative range for QPSK
        "16QAM": (20.0, 25.0),   # Mid-range for 16QAM
        "64QAM": (25.0, 30.0)    # Upper range for 64QAM
    }
    )

    @property
    def min_snr_db(self) -> float:
        """Minimum SNR in dB across all modulation schemes"""
        return min(range[0] for range in self.snr_ranges.values())

    @property
    def max_snr_db(self) -> float:
        """Maximum SNR in dB across all modulation schemes"""
        return max(range[1] for range in self.snr_ranges.values())
    
    # Modulation Configuration
    modulation_schemes: List[str] = dataclasses.field(
        default_factory=lambda: ["QPSK", "16QAM", "64QAM"]
    )

    # Dataset Generation Parameters
    total_samples: int = 1_320_000           # Total number of samples to generate
    batch_size: int = 1000                   # Processing batch size
    samples_per_modulation: int = None       # Samples per modulation scheme
    random_seed: int = 42                    # Random seed for reproducibility

    channel_dtype: tf.DType = tf.complex64
    compute_dtype: tf.DType = tf.float32

    # Performance Targets (from simulation plan)
    ber_target: float = 1e-5        # BER Target: < 10^-5 at 15 dB SNR
    sinr_target: float = 15.0       # SINR Target: > 15 dB
    spectral_efficiency_min: float = 4.0  # Minimum target: 4 bits/s/Hz
    spectral_efficiency_max: float = 8.0  # Maximum target: 8 bits/s/Hz
    throughput_target: float = 0.95         # > 95% of theoretical maximum
    # Add Sionna-specific parameters
    num_bits_per_symbol: int = 2  # For QPSK modulation
    coderate: float = 0.5         # coding rate for better error performance

    def __post_init__(self):
        """Validate and initialize dependent parameters"""
        # Calculate samples per modulation if not specified
        if self.samples_per_modulation is None:
            self.samples_per_modulation = self.total_samples // len(self.modulation_schemes)

        # Validate MIMO configuration
        assert self.num_streams <= min(self.num_tx_antennas, self.num_rx_antennas), \
            "Number of streams cannot exceed min(num_tx_antennas, num_rx_antennas)"

        # Validate SNR range
        assert self.snr_range[0] < self.snr_range[1], \
            "Invalid SNR range: min should be less than max"

        # Validate batch size
        assert self.batch_size <= self.total_samples, \
            "Batch size cannot exceed total samples"
        # Validate dtypes
        assert self.channel_dtype in [tf.complex64, tf.complex128], \
            "Channel dtype must be complex64 or complex128"
        assert self.compute_dtype in [tf.float32, tf.float64], \
            "Compute dtype must be float32 or float64"
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
        try:
            # Validate antenna configuration
            assert self.num_tx_antennas > 0, "Number of transmit antennas must be positive."
            assert self.num_rx_antennas > 0, "Number of receive antennas must be positive."
            assert 1 <= self.num_streams <= min(self.num_tx_antennas, self.num_rx_antennas), "Invalid number of streams."

            # Validate frequency
            assert 1e9 <= self.carrier_frequency <= 6e9, "Carrier frequency must be in the 1-6 GHz range."

            # Validate OFDM configuration
            assert self.num_subcarriers >= 64, "Number of subcarriers must be at least 64."
            assert self.num_ofdm_symbols > 0, "Number of OFDM symbols must be positive."
            assert self.subcarrier_spacing in [15e3, 30e3, 60e3], "Subcarrier spacing must be 15, 30, or 60 kHz."

            # Validate SNR ranges for each modulation scheme
            valid_schemes = {"QPSK", "16QAM", "64QAM"}
            for modulation, (min_snr, max_snr) in self.snr_ranges.items():
                assert modulation in valid_schemes, f"Invalid modulation scheme: {modulation}"
                assert -20 <= min_snr < max_snr <= 40, f"SNR range for {modulation} must be between -20 and 40 dB"

            # Validate overall SNR ranges
            min_snr = min(range[0] for range in self.snr_ranges.values())
            max_snr = max(range[1] for range in self.snr_ranges.values())
            assert 15 <= min_snr < max_snr <= 30, "Overall SNR range must be between 15 and 30 dB"

            # Validate element spacing
            assert 0.1 <= self.element_spacing <= 1.0, "Element spacing must be between 0.1 and 1.0 wavelengths."

            # Validate modulation schemes
            for scheme in self.modulation_schemes:
                if scheme not in valid_schemes:
                    raise ValueError(f"Invalid modulation scheme: {scheme}")
            
            # Validate performance targets
            assert 0 < self.ber_target <= 1e-4, "BER target must be positive and <= 1e-4"
            assert 10 <= self.sinr_target <= 30, "SINR target must be between 10 and 30 dB"
            assert 0 < self.spectral_efficiency_min < self.spectral_efficiency_max <= 10, "Invalid spectral efficiency range"

            # Validate Sionna-specific parameters
            assert self.num_bits_per_symbol > 0, "Number of bits per symbol must be positive"
            assert 0 < self.coderate <= 0.5, "Code rate must be between 0 and 0.5"

            # Validate batch and dataset size
            assert self.batch_size > 0, "Batch size must be positive"
            assert self.num_batches > 0, "Number of batches must be positive"

            # Validate coherence time if present
            if hasattr(self, 'coherence_time'):
                assert self.coherence_time > 0, "Coherence time must be positive"

        except AssertionError as e:
            raise ValueError(f"Parameter validation failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during parameter validation: {str(e)}")

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

