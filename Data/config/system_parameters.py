import dataclasses
import numpy as np
import tensorflow as tf
from typing import Tuple, List, Optional, Union, Dict, Any
from datetime import datetime
import json
import logging

# At the top of system_parameters.py
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available, using default memory settings")

@dataclasses.dataclass
class SystemParameters:
    # Antenna Configuration
    max_memory_fraction: float = 0.8  # Maximum fraction of GPU memory to use
    batch_size_scaling: float = 0.5   # Conservative batch size scaling
    num_tx: int = 4
    num_rx: int = 4
    num_streams: int = 4
    element_spacing: float = 0.5
    
    # Frequency Parameters
    carrier_frequency: float = 3.5e9  # 3.5 GHz
    
    # OFDM Parameters
    num_subcarriers: int = 64
    num_ofdm_symbols: int = 14
    subcarrier_spacing: float = 30e3  # 30 kHz
    
    # Channel Parameters
    num_paths: int = 20 # before number of paths was :10
    snr_range: Tuple[float, float] = (-20.0, 30.0)  # Using float values for precision
    noise_floor: float = -174  # dBm/Hz
    
    # Modulation Schemes
    modulation_schemes: List[str] = dataclasses.field(
        default_factory=lambda: ['QPSK', '16QAM', '64QAM']
    )
    
    # Path Loss Configuration
    path_loss_scenarios: List[str] = dataclasses.field(
        default_factory=lambda: ['umi', 'uma']
    )
    
    # Static User and Reproducibility Configuration
    user_mobility: bool = False  # Flag for static user scenario
    random_seed: int = 42  # Seed for reproducible results
    
    # Dataset Generation Parameters
    total_samples: int = 21_000_000
    samples_per_modulation: int = None
    replay_buffer_size: int = 21_000_000  # Add replay buffer size (this is for gpu server runing)
    

    # Add validation method
    def validate_mimo_dimensions(self):
        """Ensure MIMO dimensions are valid"""
        assert self.num_tx > 0, "Invalid number of transmit antennas"
        assert self.num_rx > 0, "Invalid number of receive antennas"
        return True

    def _initialize_hardware_parameters(self):
        try:
            
            # Get system memory
            if PSUTIL_AVAILABLE:
                system_memory_gb = psutil.virtual_memory().total / (1024**3)
                logging.info(f"Total System Memory: {system_memory_gb:.1f} GB")
            else:
                system_memory_gb = 196.0  # Default assumption
                
            # Check for GPU availability
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # Force H100 configuration for your setup
                self.batch_size = 4000  # Conservative batch size
                self.memory_threshold = 40.0  # 40GB per GPU
                self.max_batch_size = 6000  # Match with PathLossManager
                self.min_batch_size = 1000  # Conservative minimum
                
                # Configure each GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                if len(gpus) > 1:
                    self.batch_size *= 1.5  # Slight increase for multi-GPU
                    self.max_batch_size *= 1.5
                    
            else:
                # CPU configuration
                self.batch_size = 1000
                self.memory_threshold = system_memory_gb * 0.2
                self.max_batch_size = 2000
                self.min_batch_size = 500
                
            logging.info(f"Hardware Configuration:")
            logging.info(f"- Batch size: {self.batch_size}")
            logging.info(f"- Memory threshold: {self.memory_threshold:.1f} GB")
            logging.info(f"- Max batch size: {self.max_batch_size}")
            logging.info(f"- Min batch size: {self.min_batch_size}")
            
        except Exception as e:
            logging.warning(f"Hardware initialization error: {e}")
            # Conservative fallback settings
            self.batch_size = 1000
            self.memory_threshold = 4.0
            self.max_batch_size = 2000
            self.min_batch_size = 500
            
    def __post_init__(self):
        """
        Post-initialization validation and calculations
        """
        # Automatically calculate samples per modulation if not specified
        if self.samples_per_modulation is None:
            self.samples_per_modulation = self.total_samples // len(self.modulation_schemes)
        
        # Ensure num_streams does not exceed antenna count
        self.num_streams = min(self.num_streams, self.num_tx, self.num_rx)
        
        # Initialize hardware parameters
        self._initialize_hardware_parameters()
        
        # Validate parameters
        self._validate_parameters()
        
        # Set global seeds
        self.set_global_seeds()

    def __init__(
        self, 
        num_tx: int = 4,
        num_rx: int = 4,
        total_samples: int = 21_000_000,#900_000,
        **kwargs
    ):
        """
        Flexible initialization method
        """
        # Initialize default values from dataclass fields
        for field in dataclasses.fields(self):
            if isinstance(field.default, dataclasses._MISSING_TYPE):
                if field.default_factory is not dataclasses.MISSING:
                    default_value = field.default_factory()
                else:
                    default_value = None
            else:
                default_value = field.default
            object.__setattr__(self, field.name, default_value)

        # Set basic parameters
        object.__setattr__(self, 'num_tx', num_tx)
        object.__setattr__(self, 'num_rx', num_rx)
        object.__setattr__(self, 'total_samples', total_samples)

        # Set additional parameters from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                object.__setattr__(self, key, value)
            else:
                logging.warning(f"Ignoring unknown parameter: {key}")

        # Call post-initialization
        self.__post_init__()

    def _validate_parameters(self):
        """
        Rigorously validate system parameters for consistency and physical plausibility
        
        Raises:
            AssertionError: If parameters violate defined constraints
        """
        # Antenna validation
        assert self.num_tx > 0, "Number of transmit antennas must be positive"
        assert self.num_rx > 0, "Number of receive antennas must be positive"
        assert 1 <= self.num_streams <= min(self.num_tx, self.num_rx), "Invalid number of streams"
        
        # Frequency validation
        assert 1e9 <= self.carrier_frequency <= 6e9, "Carrier frequency out of realistic range"
        
        # Subcarrier and OFDM validation
        assert self.num_subcarriers > 0, "Number of subcarriers must be positive"
        assert self.num_ofdm_symbols > 0, "Number of OFDM symbols must be positive"
        
        # SNR range validation
        assert -20 <= self.snr_range[0] < self.snr_range[1] <= 40, "Unrealistic SNR range"
        
        # Element spacing validation
        assert 0.1 <= self.element_spacing <= 1.0, "Element spacing must be between 0.1 and 1.0 wavelengths"

    def set_global_seeds(self):
        """
        Comprehensive global seed setting for reproducible randomness
        
        Sets seeds for:
        - NumPy
        - TensorFlow
        - Python's random module
        - Global random state
        
        Ensures consistent randomness across different libraries
        """
        # Set seeds with error handling
        try:
            np.random.seed(self.random_seed)
            tf.random.set_seed(self.random_seed)
            
            import random
            random.seed(self.random_seed)
            
            np.random.default_rng(self.random_seed)
            
            logging.info(f"Global seeds set successfully with seed: {self.random_seed}")
        
        except Exception as e:
            logging.error(f"Could not set global seeds: {e}")
    
    def generate_reproducible_random_state(
        self, 
        additional_seed: Optional[int] = None
    ) -> np.random.RandomState:
        """
        Generate a highly reproducible and flexible random state generator
        
        Args:
            additional_seed (Optional[int]): Optional seed to mix with system seed
        
        Returns:
            numpy.random.RandomState: Reproducible random state generator
        
        Features:
        - Supports optional additional seed mixing
        - Ensures reproducibility
        - Provides flexibility for multiple random generations
        """
        # Mix system seed with additional seed if provided
        mixed_seed = (self.random_seed + (additional_seed or 0)) % (2**32 - 1)
        
        return np.random.RandomState(mixed_seed)
    
    def get_channel_generation_config(self) -> Dict[str, Any]:
        """
        Provide a comprehensive, detailed configuration for channel generation
        
        Returns:
            Dict[str, Any]: Exhaustive channel generation configuration
        
        Includes:
        - Static/dynamic user configuration
        - Detailed antenna parameters
        - Channel and resource grid settings
        - Modulation and path loss configurations
        - Generation metadata
        """
        config = {
            # Core Configuration
            'is_static': not self.user_mobility,
            'random_seed': self.random_seed,
            
            # Comprehensive Antenna Configuration
            'antenna_config': {
                'num_tx': self.num_tx,
                'num_rx': self.num_rx,
                'num_streams': self.num_streams,
                'element_spacing': self.element_spacing
            },
            
            # Detailed Channel Parameters
            'channel_parameters': {
                'carrier_frequency': self.carrier_frequency,
                'num_paths': self.num_paths,
                'snr_range': self.snr_range,
                'noise_floor': self.noise_floor
            },
            
            # Resource Grid Specifications
            'resource_grid': {
                'num_subcarriers': self.num_subcarriers,
                'num_ofdm_symbols': self.num_ofdm_symbols,
                'subcarrier_spacing': self.subcarrier_spacing
            },
            
            # Modulation and Path Loss Details
            'modulation': {
                'schemes': self.modulation_schemes,
                'path_loss_scenarios': self.path_loss_scenarios
            },
            
            # Generation Metadata
            'generation_metadata': {
                'total_samples': self.total_samples,
                'samples_per_modulation': self.samples_per_modulation,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return config
    
    def get_config_dict(self) -> Dict[str, Any]:
        """
        Generate a comprehensive, serializable configuration dictionary
        
        Returns:
            Dict[str, Any]: Fully serializable system configuration
        
        Features:
        - Handles complex types
        - Converts to JSON-serializable format
        - Adds metadata and version information
        """
        config = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            
            # Type-safe serialization
            if isinstance(value, (int, float, str, bool, type(None))):
                config[field.name] = value
            elif isinstance(value, (list, tuple)):
                config[field.name] = list(value)
            else:
                config[field.name] = str(value)
        
        # Add additional metadata
        config.update({
            'config_version': '1.1.0',
            'generation_timestamp': datetime.now().isoformat()
        })
        
        return config
    
    def update(self, **kwargs):
        """
        Dynamically update system parameters with comprehensive validation
        
        Raises:
            AttributeError: For invalid parameter updates
            ValueError: For invalid parameter values
        """
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"Invalid parameter: {key}")
            
            # Optional: Add type checking
            try:
                setattr(self, key, value)
            except ValueError as e:
                logging.error(f"Invalid value for {key}: {e}")
                raise
        
        # Re-validate parameters
        self._validate_parameters()
        self.set_global_seeds()
        
        logging.info("System parameters updated successfully")

    # Factory methods remain the same
    def create_default_system_parameters() -> "SystemParameters":
        """
        Factory method to create default system parameters
        
        Returns:
            SystemParameters: Default configuration instance
        """
        return SystemParameters()

    def create_custom_system_parameters(
        num_tx: int = 4, 
        num_rx: int = 4, 
        carrier_frequency: float = 3.5e9,
        **kwargs
    ) -> "SystemParameters":
        """
        Create custom system parameters with flexible configuration
        
        Args:
            num_tx: Number of transmit antennas
            num_rx: Number of receive antennas
            carrier_frequency: Carrier frequency in Hz
            **kwargs: Additional configuration parameters
        
        Returns:
            SystemParameters: Customized configuration instance
        """
        return SystemParameters(
            num_tx=num_tx,
            num_rx=num_rx,
            carrier_frequency=carrier_frequency,
            **kwargs
        )