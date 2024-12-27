# dataset_generator/mimo_dataset_generator.py
# Comprehensive MIMO Dataset Generation Framework
# Generates large-scale, configurable MIMO communication datasets with multiple modulation schemes
# Supports advanced channel modeling, metrics calculation, and dataset verification

import os
import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from datetime import datetime
from typing import Dict

# System configuration imports
from config.system_parameters import SystemParameters
from core.channel_model import ChannelModelManager
from core.metrics_calculator import MetricsCalculator
from core.path_loss_model import PathLossManager
from utill.logging_config import LoggerManager
from utill.tensor_shape_validator import validate_tensor_shapes
from integrity.dataset_integrity_checker import MIMODatasetIntegrityChecker

# Try to import psutil, set flag for availability
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. System memory monitoring will be limited.")

class MIMODatasetGenerator:
    """
    Comprehensive MIMO dataset generation framework
    """
    def __init__(
        self, 
        system_params: SystemParameters = None,
        logger=None,
        max_batch_size=10000  # Default maximum batch size
    ):
        """
        Initialize MIMO dataset generator
        
        Args:
            system_params (SystemParameters, optional): System configuration
            logger (logging.Logger, optional): Logger instance
            max_batch_size (int, optional): Maximum allowed batch size for processing
        """
        # Use default system parameters if not provided
        self.system_params = system_params or SystemParameters()
        self.max_batch_size = max_batch_size  # Maximum batch size parameter
        self.batch_size_scaling = 0.5  # Default scaling factor
        self.max_memory_fraction = 0.8  # Default memory fraction
        
        # Configure logger
        self.logger = logger or LoggerManager.get_logger(
            name='MIMODatasetGenerator', 
            log_level='INFO'
        )

        # Initialize batch size based on memory constraints
        self._initialize_batch_size()
        
        # Initialize core components
        self.channel_model = ChannelModelManager(self.system_params)
        self.metrics_calculator = MetricsCalculator(self.system_params)
        self.path_loss_manager = PathLossManager(self.system_params)
        self.integrity_checker = None
        
        # Initialize hardware-adaptive parameters
        self._initialize_hardware_parameters()
        
        self.validation_thresholds = {
            'eigenvalues': {'min': 1e-10, 'max': 100.0},  # Increased range
            'effective_snr': {'min': -50.0, 'max': 60.0},  # Wider SNR range
            'spectral_efficiency': {'min': 0.0, 'max': 40.0},  # Increased max
            'ber': {'min': 0.0, 'max': 1.0},  # Full BER range
            'sinr': {'min': -30.0, 'max': 50.0}  # Wider SINR range
        }

        # Set maximum batch size for GPU processing with memory monitoring
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # Configure GPU for maximum memory usage
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                # Get initial GPU memory info
                try:
                    memory_info = tf.config.experimental.get_memory_info('GPU:0')
                    total_memory = memory_info['current'] / 1e9  # Convert to GB
                    self.logger.info(f"Initial GPU memory usage: {total_memory:.2f} GB")
                    
                    # Adjust batch size based on available memory
                    if total_memory > 64.0:  # More than 64GB available
                        self.batch_size = 256_000
                    elif total_memory > 16.0:  # More than 16GB available
                        self.batch_size = 128_000
                    else:  # Limited memory
                        self.batch_size = 64_000

                except Exception as mem_error:
                    self.logger.warning(f"Could not get GPU memory info: {mem_error}")
                    self.batch_size = 5000  # Default to conservative batch size
                    
                self.logger.info(f"Using GPU with batch size {self.batch_size}")
                
                # Set up memory monitoring callback
                def memory_monitoring_callback():
                    try:
                        current_memory = tf.config.experimental.get_memory_info('GPU:0')['current'] / 1e9
                        if current_memory > 0.9 * total_memory:  # If using more than 90% memory
                            self.batch_size = max(1000, self.batch_size // 2)
                            self.logger.warning(f"High memory usage detected. Reducing batch size to {self.batch_size}")
                    except:
                        pass
                
                self.memory_callback = memory_monitoring_callback
                
            else:
                self.batch_size = 5000  # Reduced CPU batch size
                self.logger.info(f"Using CPU with batch size {self.batch_size}")
                self.memory_callback = None
                
        except tf.errors.ResourceExhaustedError as oom_error:
            self.logger.warning(f"GPU memory exhausted: {oom_error}")
            self.batch_size = self.batch_size // 2 if hasattr(self, 'batch_size') else 64000
            self.logger.info(f"Reduced batch size to {self.batch_size} due to memory constraints")
            self.memory_callback = None
            
        except Exception as e:
            self.logger.warning(f"Error configuring GPU: {e}. Defaulting to CPU processing")
            self.batch_size = 50000
            self.memory_callback = None

    def _initialize_hardware_parameters(self):
        """
        Initialize parameters optimized for high-end hardware (H100 GPUs and large RAM)
        """
        try:
            # Import psutil if available
            try:
                import psutil
                PSUTIL_AVAILABLE = True
            except ImportError:
                PSUTIL_AVAILABLE = False
                self.logger.warning("psutil not available, using default memory settings")

            # Get system memory
            if PSUTIL_AVAILABLE:
                system_memory_gb = psutil.virtual_memory().total / (1024**3)
                self.logger.info(f"Total System Memory: {system_memory_gb:.1f} GB")
            else:
                system_memory_gb = 386.0  # Default to known system memory

            # Configure GPU settings
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # Optimized settings for H100 GPUs
                self.batch_size = 32000  # Larger batch size for H100
                self.memory_threshold = 40.0  # Higher threshold for H100 (80GB per GPU)
                self.max_batch_size = 64000  # Maximum batch size
                self.min_batch_size = 16000  # Minimum batch size

                # Configure each GPU
                for gpu in gpus:
                    try:
                        # Enable memory growth for better memory management
                        tf.config.experimental.set_memory_growth(gpu, True)
                        
                        # Optional: Set memory limit per GPU (uncomment if needed)
                        # tf.config.set_logical_device_configuration(
                        #     gpu,
                        #     [tf.config.LogicalDeviceConfiguration(memory_limit=1024*75)]  # 75GB limit
                        # )
                    except RuntimeError as e:
                        self.logger.warning(f"GPU configuration warning for {gpu}: {e}")
                        continue

                # Multi-GPU optimization (for 2x H100)
                if len(gpus) > 1:
                    self.batch_size = int(self.batch_size * 1.5)  # Increase batch size for multi-GPU
                    self.max_batch_size = int(self.max_batch_size * 1.5)
                    self.logger.info(f"Multiple GPUs detected, adjusted batch size to: {self.batch_size}")

                # Enable mixed precision (crucial for H100 performance)
                try:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                except Exception as e:
                    self.logger.warning(f"Mixed precision configuration warning: {e}")

            else:
                # CPU fallback settings (utilizing high RAM)
                self.batch_size = int(system_memory_gb * 100)  # Scale with available RAM
                self.memory_threshold = system_memory_gb * 0.4  # Use 40% of system RAM
                self.max_batch_size = int(system_memory_gb * 200)
                self.min_batch_size = int(system_memory_gb * 50)

            # Advanced memory management
            self.current_memory_usage = 0.0
            self.stable_iterations = 0
            self.growth_step = 1.2
            self.memory_buffer = 0.15  # 15% memory buffer

            # Clear session and garbage collect
            tf.keras.backend.clear_session()
            import gc
            gc.collect()

            # Log detailed configuration
            self.logger.info("Hardware Configuration:")
            self.logger.info(f"- Batch size: {self.batch_size}")
            self.logger.info(f"- Memory threshold: {self.memory_threshold:.1f} GB")
            self.logger.info(f"- Max batch size: {self.max_batch_size}")
            self.logger.info(f"- Min batch size: {self.min_batch_size}")
            
            if gpus:
                self.logger.info(f"- GPU count: {len(gpus)}")
                self.logger.info(f"- Memory growth enabled: True")
                self.logger.info(f"- Mixed precision enabled: True")
                self.logger.info(f"- GPU type: H100")
                self.logger.info(f"- Available system RAM: {system_memory_gb:.1f} GB")

        except Exception as e:
            self.logger.error(f"Hardware initialization error: {e}")
            # Conservative fallback settings
            self.batch_size = 32000
            self.memory_threshold = 40.0
            self.max_batch_size = 64000
            self.min_batch_size = 16000
            
            # Ensure basic attributes are set
            if not hasattr(self, 'current_memory_usage'):
                self.current_memory_usage = 0.0
            if not hasattr(self, 'stable_iterations'):
                self.stable_iterations = 0
            if not hasattr(self, 'growth_step'):
                self.growth_step = 1.2

        # Return success status
        return gpus is not None and len(gpus) > 0

    def validate_consistency(self, f):
        """
        Validate consistency of dataset sizes across all components.
        
        Args:
            f: HDF5 file object
        
        Returns:
            bool: True if consistent, False otherwise
        """
        try:
            base_size = None
            sizes = {}
            
            # Check modulation data sizes
            for mod_scheme in f['modulation_data']:
                mod_size = f['modulation_data'][mod_scheme]['channel_response'].shape[0]
                sizes[f"modulation_{mod_scheme}"] = mod_size
                
                if base_size is None:
                    base_size = mod_size
                elif mod_size != base_size:
                    self.logger.error(f"Size mismatch in {mod_scheme}: {mod_size} vs {base_size}")
                    return False
            
            # Check path loss data sizes
            pl_size = f['path_loss_data']['fspl'].shape[0]
            sizes["path_loss"] = pl_size
            
            if pl_size != base_size:
                self.logger.error(f"Path loss data size mismatch: {pl_size} vs {base_size}")
                return False
                
            self.logger.info(f"Dataset size consistency verified: {sizes}")
            return True
        except Exception as e:
            self.logger.error(f"Consistency validation failed: {str(e)}")
            return False

    def adjust_batch_size(self, batch_idx, distances):
        """
        Adjust distances and path loss computations with updated batch size.
        
        Args:
            batch_idx: Current batch index (not used in this implementation)
            distances: Tensor containing distance values
        
        Returns:
            Adjusted FSPL tensor
        """
        try:
            batch_size = self.batch_size
            distances = distances[:batch_size]  # Ensure distances match batch size
            fspl = self.path_loss_manager.calculate_free_space_path_loss(distances)
            return fspl
        except Exception as e:
            self.logger.error(f"Batch size adjustment failed: {e}")
            raise

    def _validate_batch_data(self, batch_data: dict) -> tuple[bool, list[str]]:
        """
        Validate batch data against defined thresholds with preprocessing
        """
        errors = []
        try:
            # Preprocess data before validation
            processed_data = {}
            for key, data in batch_data.items():
                if key in self.validation_thresholds:
                    # Convert to numpy and flatten
                    data_np = data.numpy().flatten()
                    # Remove any invalid values
                    data_np = data_np[np.isfinite(data_np)]
                    processed_data[key] = data_np

            # Perform validation on processed data
            for key, threshold in self.validation_thresholds.items():
                if key in processed_data:
                    data = processed_data[key]
                    if len(data) == 0 or np.any(~np.isfinite(data)):
                        errors.append(f"{key}: contains invalid values")
                    elif np.any(data < threshold['min']) or np.any(data > threshold['max']):
                        errors.append(f"{key}: values outside threshold range [{threshold['min']}, {threshold['max']}]")
                        
            return len(errors) == 0, errors
        except Exception as e:
            self.logger.error(f"Batch validation error: {str(e)}")
            return False, [str(e)]
        
    def _prepare_output_directory(self, save_path: str):
        """
        Prepare output directory for dataset
        """
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Created output directory: {directory}")
        else:
            self.logger.info(f"Using existing output directory: {directory}")
    
    def _create_dataset_structure(self, hdf5_file, num_samples: int):
        """
        Create HDF5 dataset structure with comprehensive validation and documentation
        """
        try:
            # Validate input parameters
            if num_samples <= 0:
                raise ValueError("Number of samples must be positive")
            
            if len(self.system_params.modulation_schemes) == 0:
                raise ValueError("No modulation schemes specified")

            # Calculate samples per modulation scheme
            samples_per_mod = num_samples // len(self.system_params.modulation_schemes)
            if samples_per_mod * len(self.system_params.modulation_schemes) != num_samples:
                self.logger.warning(
                    f"Total samples {num_samples} not exactly divisible by number of "
                    f"modulation schemes {len(self.system_params.modulation_schemes)}. "
                    f"Using {samples_per_mod} samples per modulation."
                )

            # Create main groups with descriptions
            modulation_data = hdf5_file.create_group('modulation_data')
            modulation_data.attrs['description'] = 'Contains data for different modulation schemes'
            
            config_data = hdf5_file.create_group('configuration')
            config_data.attrs['description'] = 'System configuration parameters'
            
            # Create path loss data group at root level
            path_loss_data = hdf5_file.create_group('path_loss_data')
            path_loss_data.attrs['description'] = 'Path loss measurements and calculations'
            
            # Create path loss datasets at root level
            for name in ['fspl', 'scenario_pathloss']:
                dataset = path_loss_data.create_dataset(
                    name,
                    shape=(num_samples,),
                    dtype=np.float32,
                    chunks=True,
                    compression='gzip',
                    compression_opts=4
                )
                dataset.attrs['description'] = f'{name.upper()} measurements'
                dataset.attrs['units'] = 'dB'

            # Add metadata
            config_data.attrs.update({
                'creation_date': datetime.now().isoformat(),
                'total_samples': num_samples,
                'samples_per_modulation': samples_per_mod,
                'num_tx_antennas': self.system_params.num_tx,
                'num_rx_antennas': self.system_params.num_rx
            })

            # Define dataset descriptions
            dataset_descriptions = {
                'channel_response': 'Complex MIMO channel matrix response',
                'sinr': 'Signal-to-Interference-plus-Noise Ratio in dB',
                'spectral_efficiency': 'Spectral efficiency in bits/s/Hz',
                'effective_snr': 'Effective Signal-to-Noise Ratio in dB',
                'eigenvalues': 'Eigenvalues of the channel matrix',
                'ber': 'Bit Error Rate',
                'throughput': 'System throughput in bits/s',
                'inference_time': 'Processing time for inference'
            }

            # Create datasets for each modulation scheme
            for mod_scheme in self.system_params.modulation_schemes:
                # Create modulation-specific group
                mod_group = modulation_data.create_group(mod_scheme)
                mod_group.attrs['description'] = f'Data for {mod_scheme} modulation'

                # Define other datasets for this modulation
                datasets = {
                    'channel_response': {
                        'shape': (samples_per_mod, self.system_params.num_rx, self.system_params.num_tx),
                        'dtype': np.complex64
                    },
                    'sinr': {
                        'shape': (samples_per_mod,),
                        'dtype': np.float32
                    },
                    'spectral_efficiency': {
                        'shape': (samples_per_mod,),
                        'dtype': np.float32
                    },
                    'effective_snr': {
                        'shape': (samples_per_mod,),
                        'dtype': np.float32
                    },
                    'eigenvalues': {
                        'shape': (samples_per_mod, self.system_params.num_rx),
                        'dtype': np.float32
                    },
                    'ber': {
                        'shape': (samples_per_mod,),
                        'dtype': np.float32
                    },
                    'throughput': {
                        'shape': (samples_per_mod,),
                        'dtype': np.float32
                    },
                    'inference_time': {
                        'shape': (samples_per_mod,),
                        'dtype': np.float32
                    }
                }
                
                # Create datasets with descriptions and chunking
                for name, config in datasets.items():
                    dataset = mod_group.create_dataset(
                        name,
                        shape=config['shape'],
                        dtype=config['dtype'],
                        chunks=True,
                        compression='gzip',
                        compression_opts=4
                    )
                    dataset.attrs['description'] = dataset_descriptions[name]
                    dataset.attrs['units'] = self._get_dataset_units(name)

            self.logger.info("Dataset structure created successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to create dataset structure: {str(e)}")
            raise

    def generate_dataset(self, num_samples: int, save_path: str = 'dataset/mimo_dataset.h5'):
        """
        Generate comprehensive MIMO dataset with enhanced shape validation and memory management
        """
        failed_batches = []  # Initialize the list for failed batches
        try:

            self._prepare_output_directory(save_path)

            # Check and remove existing file
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                    self.logger.info(f"Removed existing dataset file: {save_path}")
                except OSError as e:
                    self.logger.error(f"Error removing existing file: {e}")
                    raise

            # Initial batch size safety check
            self.batch_size = self._check_batch_size_safety(self.batch_size)
            self.logger.info(f"Initial batch size adjusted for memory safety: {self.batch_size}")

            samples_per_mod = num_samples // len(self.system_params.modulation_schemes)
            
            if num_samples % len(self.system_params.modulation_schemes) != 0:
                self.logger.warning(
                    f"Total samples {num_samples} not exactly divisible by number of "
                    f"modulation schemes {len(self.system_params.modulation_schemes)}. "
                    f"Using {samples_per_mod} samples per modulation."
                )
            
            # Open file in write mode after ensuring it doesn't exist
            with h5py.File(save_path, 'w') as f:
                self._create_dataset_structure(f, num_samples)
                
                # Initialize path loss offset counter
                path_loss_offset = 0  
                
                total_progress = tqdm(total=num_samples, desc="Total Dataset Generation", unit="samples")
                
                for mod_scheme in self.system_params.modulation_schemes:
                    mod_group = f['modulation_data'][mod_scheme]
                    mod_progress = tqdm(
                        total=samples_per_mod,
                        desc=f"{mod_scheme} Generation",
                        unit="samples",
                        leave=False
                    )
                    
                    batch_idx = 0
                    while batch_idx < samples_per_mod // self.batch_size:
                        try:
                            start_idx = batch_idx * self.batch_size
                            end_idx = start_idx + self.batch_size
                            
                            # Generate input data
                            # Generate input data with physically meaningful minimum distance
                            distances = tf.random.uniform(
                                [self.batch_size], 
                                minval=1.0,  # Minimum distance (1 meter)
                                maxval=500.0  # Maximum distance (500 meters)
)
                            snr_db = tf.random.uniform(
                                [self.batch_size],
                                self.system_params.snr_range[0],
                                self.system_params.snr_range[1]
                            )

                            # Generate channel data with error handling
                            try:
                                channel_data = self.channel_model.generate_mimo_channel(self.batch_size, snr_db)
                                h_perfect = channel_data['perfect_channel']
                                h_noisy = channel_data['noisy_channel']
                            except tf.errors.ResourceExhaustedError:
                                self.batch_size = max(1000, self.batch_size // 2)
                                self.logger.warning(f"OOM error in channel generation, reducing batch size to {self.batch_size}")
                                self._manage_memory()
                                continue

                            # Ensure correct shape [batch_size, num_rx, num_tx]
                            h_perfect = tf.reshape(h_perfect, 
                                                [self.batch_size, self.system_params.num_rx, self.system_params.num_tx])

                            # Calculate and apply path loss
                            path_loss_db = self.path_loss_manager.calculate_path_loss(
                                distances[:self.batch_size], 'umi'
                            )
                            path_loss_db = tf.reshape(path_loss_db[:self.batch_size], [self.batch_size])
                            path_loss_linear = tf.pow(10.0, -path_loss_db / 20.0)
                            path_loss_shaped = tf.reshape(path_loss_linear, [self.batch_size, 1, 1])
                            h_with_pl = h_perfect * tf.cast(path_loss_shaped, dtype=h_perfect.dtype)
                            
                            # Generate and process symbols
                            tx_symbols = self.channel_model.generate_qam_symbols(self.batch_size, mod_scheme)
                            tx_symbols = tf.reshape(tx_symbols, [self.batch_size, self.system_params.num_tx, 1])
                            rx_symbols = tf.matmul(h_with_pl, tx_symbols)
                            
                            # Calculate all metrics first
                            metrics = self.metrics_calculator.calculate_performance_metrics(
                                h_with_pl,
                                tx_symbols,
                                rx_symbols,
                                snr_db
                            )
                            
                            # Calculate enhanced metrics
                            self.metrics_calculator.set_current_modulation(mod_scheme)
                            enhanced_metrics = self.metrics_calculator.calculate_enhanced_metrics(
                                h_with_pl, tx_symbols, rx_symbols, snr_db
                            )
                            
                            # Process metrics with explicit shape control
                            effective_snr = tf.reshape(tf.squeeze(metrics['effective_snr']), [self.batch_size])
                            spectral_efficiency = tf.reshape(tf.squeeze(metrics['spectral_efficiency']), [self.batch_size])
                            sinr = tf.reshape(tf.squeeze(metrics['sinr']), [self.batch_size])
                            eigenvalues = metrics['eigenvalues']
                            
                            # Create batch data after all metrics are calculated
                            batch_data = {
                                'eigenvalues': eigenvalues,
                                'effective_snr': effective_snr,
                                'spectral_efficiency': spectral_efficiency,
                                'sinr': sinr,
                                'ber': enhanced_metrics['ber']
                            }

                            # Validate batch data
                            MAX_RETRIES = 3
                            retry_count = 0
                            while retry_count < MAX_RETRIES:
                                is_valid, validation_errors = self._validate_batch_data(batch_data)
                                if is_valid:
                                    break
                                
                                retry_count += 1
                                self.logger.warning(f"Invalid batch at index {batch_idx} (attempt {retry_count}/{MAX_RETRIES}):")
                                for error in validation_errors:
                                    self.logger.warning(f"  - {error}")

                            if retry_count == MAX_RETRIES:
                                self.logger.error(f"Failed to generate valid batch after {MAX_RETRIES} attempts. Logging failed batch data.")
                                failed_batches.append(batch_data)  # Maintain a list of failed batches
                                continue

                            # Before calculating path loss
                            self.logger.info(f"\nProcessing batch for {mod_scheme}:")
                            self.logger.info(f"Batch index: {batch_idx}, Range: {start_idx}-{end_idx}")
                            self.logger.info(f"Distances shape: {distances.shape}")

                            # Calculate path loss data
                            # Align path loss dataset sizes with modulation datasets
                            total_samples = len(f['modulation_data'][mod_scheme]['effective_snr'])  # Base size
                            mod_data_size = f['modulation_data'][mod_scheme]['effective_snr'].shape[0]
                            fspl = self.path_loss_manager.calculate_free_space_path_loss(distances[:mod_data_size])
                            scenario_pl = self.path_loss_manager.calculate_path_loss(distances[:mod_data_size], 'umi')
                            
                            # Debug checks before saving
                            fspl_stats = {
                                'min': tf.reduce_min(fspl).numpy(),
                                'max': tf.reduce_max(fspl).numpy(),
                                'mean': tf.reduce_mean(fspl).numpy(),
                                'zeros': tf.reduce_sum(tf.cast(tf.equal(fspl, 0.0), tf.int32)).numpy()
                            }
                            
                            self.logger.info(
                                f"\nPath Loss Stats before saving:"
                                f"\n - FSPL range: [{fspl_stats['min']:.2f}, {fspl_stats['max']:.2f}] dB"
                                f"\n - FSPL mean: {fspl_stats['mean']:.2f} dB"
                                f"\n - Zero values: {fspl_stats['zeros']}"
                            )

                                                        # Add enhanced validation and safety checks
                            min_pl = 20.0
                            max_pl = 160.0
                            # Clip values with validation
                            fspl = tf.clip_by_value(fspl, min_pl, max_pl)
                            scenario_pl = tf.clip_by_value(scenario_pl, min_pl, max_pl)

                            # Add validation checks with detailed logging
                            any_invalid_fspl = tf.math.logical_or(
                                tf.math.less(fspl, min_pl),
                                tf.math.greater(fspl, max_pl)
                            )
                            any_invalid_scenario = tf.math.logical_or(
                                tf.math.less(scenario_pl, min_pl),
                                tf.math.greater(scenario_pl, max_pl)
                            )

                            if tf.math.reduce_any(any_invalid_fspl):
                                min_val = tf.reduce_min(fspl)
                                max_val = tf.reduce_max(fspl)
                                self.logger.warning(
                                    f"FSPL values required clipping. Range before clip: [{min_val:.2f}, {max_val:.2f}] dB"
                                )

                            if tf.math.reduce_any(any_invalid_scenario):
                                min_val = tf.reduce_min(scenario_pl)
                                max_val = tf.reduce_max(scenario_pl)
                                self.logger.warning(
                                    f"Scenario path loss values required clipping. Range before clip: [{min_val:.2f}, {max_val:.2f}] dB"
                                )

                            # Ensure no NaN or infinite values
                            fspl = tf.where(tf.math.is_finite(fspl), fspl, min_pl)
                            scenario_pl = tf.where(tf.math.is_finite(scenario_pl), scenario_pl, min_pl)

                            # Add logging for clipped values
                            if tf.reduce_any(fspl < 20.0) or tf.reduce_any(fspl > 160.0):
                                self.logger.warning("Some FSPL values were clipped to physical bounds")
                            if tf.reduce_any(scenario_pl < 20.0) or tf.reduce_any(scenario_pl > 160.0):
                                self.logger.warning("Some scenario path loss values were clipped to physical bounds")

                            # Slice the tensors to match batch_size
                            fspl = tf.slice(fspl, [0], [self.batch_size])
                            scenario_pl = tf.slice(scenario_pl, [0], [self.batch_size])

                            # Save all data to HDF5
                            mod_group['channel_response'][start_idx:end_idx] = h_perfect.numpy()
                            mod_group['sinr'][start_idx:end_idx] = sinr.numpy()
                            mod_group['spectral_efficiency'][start_idx:end_idx] = spectral_efficiency.numpy()
                            mod_group['effective_snr'][start_idx:end_idx] = effective_snr.numpy()
                            mod_group['eigenvalues'][start_idx:end_idx] = eigenvalues.numpy()
                            mod_group['ber'][start_idx:end_idx] = enhanced_metrics['ber']
                            mod_group['throughput'][start_idx:end_idx] = enhanced_metrics['throughput']
                            
                            # Calculate global indices for path loss data
                            pl_start_idx = path_loss_offset + start_idx
                            pl_end_idx = path_loss_offset + end_idx

                            # Save path loss data with global indexing
                            f['path_loss_data']['fspl'][pl_start_idx:pl_end_idx] = fspl.numpy()
                            f['path_loss_data']['scenario_pathloss'][pl_start_idx:pl_end_idx] = scenario_pl.numpy()
                            
                            # Add verification logging
                            self.logger.info(
                                f"\nVerification after save:"
                                f"\n - Saved values range: [{np.min(f['path_loss_data']['fspl'][start_idx:end_idx]):.2f}, "
                                f"{np.max(f['path_loss_data']['fspl'][start_idx:end_idx]):.2f}] dB"
                                f"\n - Zero values in saved data: {np.sum(f['path_loss_data']['fspl'][start_idx:end_idx] == 0.0)}"
                            )

                            # Update progress
                            mod_progress.update(self.batch_size)
                            total_progress.update(self.batch_size)
                            
                            # Memory management
                            self._manage_memory()
                            
                            # Increment batch index
                            batch_idx += 1
                            
                        except tf.errors.ResourceExhaustedError as oom_error:
                            self.batch_size = max(1000, self.batch_size // 2)
                            self.logger.warning(f"OOM error detected, reducing batch size to {self.batch_size}")
                            self._manage_memory()
                            continue
                            
                        except Exception as batch_error:
                            self.logger.error(f"Error processing batch {batch_idx}: {str(batch_error)}")
                            if isinstance(batch_error, tf.errors.ResourceExhaustedError):
                                self.batch_size = max(1000, self.batch_size // 2)
                                self.logger.warning(f"Reducing batch size to {self.batch_size} due to memory constraints")
                                continue
                            else:
                                raise
                    
                    path_loss_offset += end_idx - start_idx # Use actual processed range
                    
                    mod_progress.close()
                
                total_progress.close()

                # Add final metadata
                f.attrs['completion_date'] = datetime.now().isoformat()
                f.attrs['total_samples_generated'] = num_samples
                f.attrs['final_batch_size'] = self.batch_size
                
                # Verify dataset integrity
                self.integrity_checker = MIMODatasetIntegrityChecker(save_path)
                integrity_report = self.integrity_checker.check_dataset_integrity()
                
                if integrity_report['overall_status']:
                    self.logger.info("Dataset integrity verification passed")
                    return True
                else:
                    self.logger.error("Dataset integrity verification failed")
                    self.logger.error("Integrity report:", integrity_report)
                    return False

        except Exception as e:
            self.logger.error(f"Dataset generation failed: {str(e)}")
            self.logger.error("Detailed error traceback:", exc_info=True)
            raise

        
    def _process_batch(self, batch_size: int, mod_scheme: str) -> Dict[str, tf.Tensor]:
        """
        Process a single batch of data
        
        Args:
            batch_size: Size of the batch to process
            mod_scheme: Current modulation scheme
            
        Returns:
            Dictionary containing processed batch data
        """
        try:
            # Generate channel data
            channel_data = self.channel_model.generate_mimo_channel(
                batch_size,
                self.system_params.snr_range
            )
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_performance_metrics(
                channel_data['perfect_channel'],
                channel_data['tx_symbols'],
                channel_data['rx_symbols'],
                channel_data['snr_db']
            )
            
            return {
                'channel_response': channel_data['perfect_channel'],
                'sinr': metrics['sinr'],
                'spectral_efficiency': metrics['spectral_efficiency'],
                'effective_snr': metrics['effective_snr'],
                'eigenvalues': metrics['eigenvalues']
            }
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            raise

    def _get_dataset_units(self, dataset_name: str) -> str:
        """Helper method to return appropriate units for each dataset type"""
        units_map = {
            'channel_response': 'complex',
            'sinr': 'dB',
            'spectral_efficiency': 'bits/s/Hz',
            'effective_snr': 'dB',
            'eigenvalues': 'linear',
            'ber': 'ratio',
            'throughput': 'bits/s',
            'inference_time': 'seconds'
        }
        return units_map.get(dataset_name, 'dimensionless')

    def safe_memory_growth(self):
        """
        Simplified memory growth for VM environment
        """
        try:
            if hasattr(self, 'stable_iterations'):
                self.stable_iterations += 1
                
                if self.stable_iterations >= 10:
                    current_batch = self.batch_size
                    potential_batch = min(int(current_batch * 1.2), self.max_batch_size)
                    
                    if potential_batch > current_batch:
                        self.batch_size = potential_batch
                        self.logger.info(f"Increasing batch size to {self.batch_size}")
                        
                    self.stable_iterations = 0
                    
        except Exception as e:
            self.logger.warning(f"Memory growth check failed: {e}")

    def verify_dataset(self, save_path: str) -> bool:
        """
        Verify dataset integrity using MIMODatasetIntegrityChecker
        """
        try:
            with h5py.File(save_path, 'r') as f:
                # First verify basic structure
                if 'modulation_data' not in f:
                    self.logger.error("Missing modulation_data group")
                    return False

                # Check each modulation scheme
                for mod_scheme in self.system_params.modulation_schemes:
                    if mod_scheme not in f['modulation_data']:
                        self.logger.error(f"Missing data for modulation scheme: {mod_scheme}")
                        return False

                    mod_group = f['modulation_data'][mod_scheme]
                    required_datasets = [
                        'channel_response', 'sinr', 'spectral_efficiency', 
                        'effective_snr', 'eigenvalues', 'ber', 'throughput'
                    ]

                    # Verify all required datasets exist and have data
                    for dataset_name in required_datasets:
                        if dataset_name not in mod_group:
                            self.logger.error(f"Missing dataset {dataset_name} for {mod_scheme}")
                            return False
                        
                        dataset = mod_group[dataset_name]
                        if dataset.shape[0] == 0:
                            self.logger.error(f"Empty dataset {dataset_name} for {mod_scheme}")
                            return False

                        # Basic statistical checks
                        try:
                            data = dataset[:]
                            stats = {
                                'mean': np.mean(np.abs(data)),
                                'std': np.std(np.abs(data)),
                                'min': np.min(np.abs(data)),
                                'max': np.max(np.abs(data))
                            }
                            self.logger.info(f"{mod_scheme} - {dataset_name} statistics: {stats}")
                        except Exception as e:
                            self.logger.error(f"Error computing statistics for {dataset_name}: {str(e)}")
                            return False

                # Verify path loss data
                if 'path_loss_data' not in f:
                    self.logger.error("Missing path_loss_data group")
                    return False

                for dataset_name in ['fspl', 'scenario_pathloss']:
                    if dataset_name not in f['path_loss_data']:
                        self.logger.error(f"Missing path loss dataset: {dataset_name}")
                        return False

                self.logger.info("Dataset verification successful")
                return True

        except Exception as e:
            self.logger.error(f"Dataset verification error: {str(e)}")
            return False
        
    # In mimo_dataset_generator.py, enhance memory management:
    def _manage_memory(self):
        """
        Enhanced memory management for H100 GPUs
        """
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                return
                
            # Monitor each GPU
            for gpu_idx, gpu in enumerate(gpus):
                try:
                    memory_info = tf.config.experimental.get_memory_info(f'GPU:{gpu_idx}')
                    current_usage = memory_info['current'] / 1e9
                    self.logger.info(f"GPU:{gpu_idx} memory usage: {current_usage:.2f} GB")
                    
                    # Adjust batch size based on memory usage
                    if current_usage > self.memory_threshold * 0.9:  # Over 90% threshold
                        new_batch_size = max(self.min_batch_size, self.batch_size // 2)
                        if new_batch_size != self.batch_size:
                            self.logger.warning(f"High memory usage. Reducing batch size from {self.batch_size} to {new_batch_size}")
                            self.batch_size = new_batch_size
                            self.stable_iterations = 0
                            
                            # Force cleanup
                            tf.keras.backend.clear_session()
                            import gc
                            gc.collect()
                            
                except Exception as e:
                    self.logger.warning(f"Error monitoring GPU:{gpu_idx} - {str(e)}")
            
            # Try to grow batch size if conditions are good
            self.safe_memory_growth()
                
        except Exception as e:
            self.logger.error(f"Memory management failed: {str(e)}")
            # Fall back to conservative batch size
            self.batch_size = self.min_batch_size


    def _check_memory_requirements(self, batch_size: int) -> bool:
        """
        Check if system has enough memory for the requested batch size
        """
        try:
            # Estimate memory requirement (rough calculation)
            sample_size = self.system_params.num_tx * self.system_params.num_rx * 8  # 8 bytes per complex64
            total_size = batch_size * sample_size * 3  # Factor of 3 for various tensors
            
            # Get available system memory
            import psutil
            available_memory = psutil.virtual_memory().available
            
            return total_size < available_memory * 0.7  # Keep 30% memory buffer for 64GB system
        except:
            return True  # Default to True if unable to check
        

    def memory_monitoring_callback(self):
        """Memory monitoring callback method"""
        try:
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            current_memory = memory_info['current'] / 1e9
            total_memory = memory_info['peak'] / 1e9  # Use peak memory as total available
            
            if current_memory > 0.9 * total_memory:  # If using more than 90% memory
                new_batch_size = max(64000, self.batch_size // 2)  # Minimum 64000 for 64GB RAM
                self.logger.warning(f"High memory usage detected. Reducing batch size to {new_batch_size}")
                self.batch_size = new_batch_size
        except Exception as e:
            self.logger.debug(f"Memory monitoring error: {e}")
            pass
    
    def _check_batch_size_safety(self, batch_size: int) -> int:
        """
        Verify and adjust batch size based on memory constraints and PathLossManager settings.
        """
        try:
            # Get max_batch_size from PathLossManager or default
            max_batch_size = getattr(self.path_loss_manager.system_params, 'max_batch_size', self.max_batch_size)
            if batch_size > max_batch_size:
                self.logger.warning(
                    f"Batch size {batch_size} exceeds max allowed by PathLossManager ({max_batch_size}). "
                    f"Reducing to {max_batch_size}."
                )
                batch_size = max_batch_size

            # Further adjustments based on system memory
            if PSUTIL_AVAILABLE:
                available_memory = psutil.virtual_memory().available / (1024**3)  # GB
                sample_size = self.system_params.num_tx * self.system_params.num_rx * 8 * 3  # Bytes per sample
                memory_per_batch = (batch_size * sample_size) / (1024**3)  # Convert to GB

                if memory_per_batch > available_memory * 0.4:  # Adjust batch size if exceeding 40% of memory
                    new_batch_size = int((available_memory * 0.4 * 1024**3) / sample_size)
                    new_batch_size = (new_batch_size // 1000) * 1000  # Round to nearest 1000
                    self.logger.warning(
                        f"Batch size {batch_size} exceeds memory constraints. "
                        f"Reducing to {new_batch_size}."
                    )
                    return max(1000, new_batch_size)

            return batch_size

        except Exception as e:
            self.logger.warning(f"Error in batch size safety check: {e}")
            return min(batch_size, self.max_batch_size)


        
# Example usage
def main():
    generator = MIMODatasetGenerator()
    generator.generate_dataset(num_samples=20_000_000)
    generator.verify_dataset('dataset/mimo_dataset.h5')

if __name__ == "__main__":
    main()