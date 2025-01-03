# dataset_generator/mimo_dataset_generator.py
# Comprehensive MIMO Dataset Generation Framework
# Generates large-scale, configurable MIMO communication datasets with multiple modulation schemes
# Supports advanced channel modeling, metrics calculation, and dataset verification
# Tensor Dimensionality:
# - Ensure tensors match the expected shapes at all stages.
# - Typical channel response shape: (Batch Size, Num RX Antennas, Num TX Antennas).
# - Validate tensor dimensions before reshaping or matrix operations.

import os
from datetime import datetime
from typing import Dict
import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from config.system_parameters import SystemParameters
from core.channel_model import ChannelModelManager
from core.metrics_calculator import MetricsCalculator
from core.path_loss_model import PathLossManager
from utill.logging_config import LoggerManager
from utill.tensor_shape_validator import (
    validate_tensor_shapes,
    validate_complex_tensor,
    verify_batch_consistency
)
from integrity.dataset_integrity_checker import MIMODatasetIntegrityChecker

# Try to import psutil, set flag for availability
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. System memory monitoring will be limited.")

class MIMODatasetGenerator:
    __version__ = '2.0.0'
    """
    Comprehensive MIMO dataset generation framework
    """
    def __init__(
        self, 
        system_params: SystemParameters = None,
        logger=None,
        max_batch_size=10000
    ):
        """
        Initialize MIMO dataset generator
        
        Args:
            system_params (SystemParameters, optional): System configuration
            logger (logging.Logger, optional): Logger instance
            max_batch_size (int, optional): Maximum allowed batch size for processing
        """
        # Configure logger first
        self.logger = logger or LoggerManager.get_logger(
            name='MIMODatasetGenerator', 
            log_level='INFO'
        )

        # Use default system parameters if not provided
        self.system_params = system_params or SystemParameters()
        
        # Initialize core components before batch size checks
        self.path_loss_manager = PathLossManager(self.system_params)
        self.channel_model = ChannelModelManager(self.system_params)
        self.metrics_calculator = MetricsCalculator(self.system_params)
        self.integrity_checker = None
        
        # Initialize batch size parameters
        self.batch_size_scaling = 0.5  # Default scaling factor
        self.max_memory_fraction = 0.8  # Default memory fraction
        self.max_batch_size = max_batch_size
        
        # Now check and initialize batch size
        self.batch_size = self._check_batch_size(max_batch_size)
        self._initialize_batch_size()
        
        # Initialize hardware parameters last
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

    def _initialize_batch_size(self):
        """
        Initialize batch size based on memory and system constraints with improved memory estimation.
        """
        try:
            if PSUTIL_AVAILABLE:
                # Get available system memory in GB
                available_memory = psutil.virtual_memory().available / (1024**3)
                
                # Calculate memory requirements per sample
                sample_memory = (
                    self.system_params.num_rx * 
                    self.system_params.num_tx * 
                    8  # Complex64 = 8 bytes
                ) / (1024**3)  # Convert to GB
                
                # Calculate safe batch size (using 70% of available memory)
                safe_memory = available_memory * 0.7
                calculated_batch_size = int(safe_memory / sample_memory)
                
                # Apply constraints
                min_batch_size = 1000
                max_batch_size = 32000
                
                self.batch_size = max(min_batch_size, 
                                    min(calculated_batch_size, max_batch_size))
                
                self.logger.info(
                    f"Memory-based batch size calculation:"
                    f"\n - Available Memory: {available_memory:.2f} GB"
                    f"\n - Sample Memory: {sample_memory:.6f} GB"
                    f"\n - Calculated Batch Size: {self.batch_size}"
                )
            else:
                self.batch_size = 4000  # Default conservative batch size
                self.logger.warning("psutil not available, using default batch size")
                
        except Exception as e:
            self.logger.error(f"Error during batch size initialization: {str(e)}")
            self.batch_size = 1000  # Fallback to minimal batch size


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
        Validate batch data against defined thresholds with enhanced validation and preprocessing.
        
        Args:
            batch_data (dict): Dictionary containing batch data to validate
            
        Returns:
            tuple[bool, list[str]]: Validation status and list of error messages
        """
        errors = []
        try:
            # Define validation thresholds if not already defined
            self.validation_thresholds = {
                'path_loss': {'min': 20.0, 'max': 160.0},
                'sinr': {'min': -20.0, 'max': 30.0},
                'spectral_efficiency': {'min': 0.0, 'max': 40.0},
                'effective_snr': {'min': -10.0, 'max': 40.0},
                'channel_response': {'min': -100.0, 'max': 100.0}
            }

            # Preprocess and validate data
            for key, data in batch_data.items():
                if key in self.validation_thresholds:
                    try:
                        # Convert to numpy and handle complex data
                        if tf.is_tensor(data):
                            if data.dtype.is_complex:
                                data_np = np.abs(data.numpy()).flatten()
                            else:
                                data_np = data.numpy().flatten()
                        else:
                            data_np = np.asarray(data).flatten()

                        # Basic data quality checks
                        if len(data_np) == 0:
                            errors.append(f"{key}: Empty data")
                            continue

                        if np.any(~np.isfinite(data_np)):
                            errors.append(f"{key}: Contains NaN or Inf values")
                            continue

                        # Threshold validation
                        threshold = self.validation_thresholds[key]
                        stats = {
                            'min': np.min(data_np),
                            'max': np.max(data_np),
                            'mean': np.mean(data_np),
                            'std': np.std(data_np)
                        }

                        if stats['min'] < threshold['min'] or stats['max'] > threshold['max']:
                            errors.append(
                                f"{key}: Values outside threshold range "
                                f"[{threshold['min']}, {threshold['max']}]. "
                                f"Stats: {stats}"
                            )

                    except Exception as val_error:
                        errors.append(f"{key}: Validation error - {str(val_error)}")

            # Log validation results
            if errors:
                self.logger.warning(f"Batch validation failed with {len(errors)} errors")
                for error in errors:
                    self.logger.debug(f"Validation error: {error}")
            else:
                self.logger.debug("Batch validation successful")

            return len(errors) == 0, errors

        except Exception as e:
            self.logger.error(f"Batch validation error: {str(e)}")
            return False, [f"Critical validation error: {str(e)}"]
            
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
        Create HDF5 dataset structure with comprehensive validation and documentation.
        
        Args:
            hdf5_file: HDF5 file object
            num_samples (int): Total number of samples to generate
            
        Raises:
            ValueError: If input parameters are invalid
        """
        try:
            # Input validation with detailed messages
            if not isinstance(num_samples, int) or num_samples <= 0:
                raise ValueError(f"Invalid num_samples: {num_samples}. Must be positive integer.")
            
            if not self.system_params.modulation_schemes:
                raise ValueError("No modulation schemes specified in system parameters.")

            # Calculate samples distribution
            samples_per_mod = num_samples // len(self.system_params.modulation_schemes)
            if samples_per_mod * len(self.system_params.modulation_schemes) != num_samples:
                self.logger.warning(
                    f"Adjusting total samples from {num_samples} to "
                    f"{samples_per_mod * len(self.system_params.modulation_schemes)} "
                    f"for even distribution across {len(self.system_params.modulation_schemes)} "
                    "modulation schemes."
                )

            # Create main groups with enhanced metadata
            groups = {
                'modulation_data': 'MIMO channel and performance metrics for each modulation',
                'path_loss_data': 'Path loss measurements and environmental factors',
                'configuration': 'System configuration and dataset parameters',
                'metadata': 'Dataset creation and validation information'
            }
            
            created_groups = {
                name: hdf5_file.create_group(name) for name in groups
            }
            
            for name, group in created_groups.items():
                group.attrs['description'] = groups[name]
                group.attrs['creation_time'] = datetime.now().isoformat()

            # Enhanced configuration metadata
            config_metadata = {
                'creation_date': datetime.now().isoformat(),
                'total_samples': num_samples,
                'samples_per_modulation': samples_per_mod,
                'num_tx_antennas': self.system_params.num_tx,
                'num_rx_antennas': self.system_params.num_rx,
                'carrier_frequency': self.system_params.carrier_frequency,
                'bandwidth': self.system_params.bandwidth,
                'snr_range': self.system_params.snr_range,
                'modulation_schemes': list(self.system_params.modulation_schemes),
                'dataset_version': '2.0',
                'generator_version': self.__class__.__version__
            }
            
            for key, value in config_metadata.items():
                created_groups['configuration'].attrs[key] = value

            # Define dataset configurations with enhanced metadata
            dataset_configs = {
                'channel_response': {
                    'shape': (samples_per_mod, self.system_params.num_rx, self.system_params.num_tx),
                    'dtype': np.complex64,
                    'description': 'Complex MIMO channel matrix response',
                    'units': 'linear',
                    'valid_range': [-100, 100]
                },
                'sinr': {
                    'shape': (samples_per_mod,),
                    'dtype': np.float32,
                    'description': 'Signal-to-Interference-plus-Noise Ratio',
                    'units': 'dB',
                    'valid_range': [-20, 30]
                },
                'spectral_efficiency': {
                    'shape': (samples_per_mod,),
                    'dtype': np.float32,
                    'description': 'Spectral efficiency',
                    'units': 'bits/s/Hz',
                    'valid_range': [0, 40]
                },
                'path_loss': {
                    'shape': (samples_per_mod,),
                    'dtype': np.float32,
                    'description': 'Path loss including shadowing',
                    'units': 'dB',
                    'valid_range': [20, 160]
                }
            }

            # Create datasets for each modulation scheme with enhanced structure
            for mod_scheme in self.system_params.modulation_schemes:
                mod_group = created_groups['modulation_data'].create_group(mod_scheme)
                mod_group.attrs.update({
                    'description': f'Data for {mod_scheme} modulation',
                    'modulation_order': self._get_modulation_order(mod_scheme),
                    'theoretical_max_spectral_efficiency': self._calculate_max_spectral_efficiency(mod_scheme)
                })

                # Create datasets with enhanced metadata and validation
                for name, config in dataset_configs.items():
                    dataset = mod_group.create_dataset(
                        name,
                        shape=config['shape'],
                        dtype=config['dtype'],
                        chunks=True,
                        compression='gzip',
                        compression_opts=4
                    )
                    
                    # Enhanced dataset attributes
                    dataset.attrs.update({
                        'description': config['description'],
                        'units': config['units'],
                        'valid_range': config['valid_range'],
                        'creation_time': datetime.now().isoformat(),
                        'statistics': {
                            'mean': 0.0,
                            'std': 0.0,
                            'min': float('inf'),
                            'max': float('-inf')
                        }
                    })

            self.logger.info(
                f"Created dataset structure with {len(self.system_params.modulation_schemes)} "
                f"modulation schemes and {samples_per_mod} samples per modulation"
            )
            
        except Exception as e:
            self.logger.error(f"Dataset structure creation failed: {str(e)}")
            raise

    def _get_modulation_order(self, mod_scheme: str) -> int:
        """Get modulation order for given scheme."""
        modulation_orders = {
            'BPSK': 2, 'QPSK': 4, '16QAM': 16, '64QAM': 64, '256QAM': 256
        }
        return modulation_orders.get(mod_scheme, 0)

    def _calculate_max_spectral_efficiency(self, mod_scheme: str) -> float:
        """Calculate theoretical maximum spectral efficiency."""
        order = self._get_modulation_order(mod_scheme)
        return self.system_params.num_tx * np.log2(order) if order > 0 else 0.0

    def _save_batch_to_hdf5(self, f, mod_group, batch_data, start_idx, end_idx, path_loss_offset):
        """Save batch data to HDF5 file"""
        try:
            for key, data in batch_data.items():
                if key in mod_group:
                    mod_group[key][start_idx:end_idx] = data.numpy()
                elif key in ['path_loss', 'distances']:
                    pl_start = path_loss_offset + start_idx
                    pl_end = path_loss_offset + end_idx
                    f['path_loss_data'][key][pl_start:pl_end] = data.numpy()
        except Exception as e:
            self.logger.error(f"Failed to save batch to HDF5: {str(e)}")
            raise

    def generate_dataset(self, num_samples: int, save_path: str = 'dataset/mimo_dataset.h5'):
        """
        Generate comprehensive MIMO dataset with enhanced validation and monitoring
        
        Args:
            num_samples (int): Total number of samples to generate
            save_path (str): Path to save the HDF5 dataset
            
        Returns:
            bool: True if generation successful, False otherwise
        """
        generation_stats = {
            'successful_batches': 0,
            'failed_batches': 0,
            'total_samples_generated': 0
        }
        
        try:
            # Prepare output directory and validate inputs
            self._prepare_output_directory(save_path)
            if os.path.exists(save_path):
                os.remove(save_path)
                self.logger.info(f"Removed existing dataset file: {save_path}")

            # Calculate samples per modulation scheme
            samples_per_mod = num_samples // len(self.system_params.modulation_schemes)
            if num_samples % len(self.system_params.modulation_schemes) != 0:
                self.logger.warning(
                    f"Adjusting total samples from {num_samples} to "
                    f"{samples_per_mod * len(self.system_params.modulation_schemes)}"
                )

            with h5py.File(save_path, 'w') as f:
                # Create dataset structure
                self._create_dataset_structure(f, num_samples)
                path_loss_offset = 0

                # Progress tracking
                with tqdm(total=num_samples, desc="Total Progress") as total_progress:
                    for mod_scheme in self.system_params.modulation_schemes:
                        mod_group = f['modulation_data'][mod_scheme]
                        
                        with tqdm(total=samples_per_mod, 
                                desc=f"{mod_scheme}", 
                                leave=False) as mod_progress:
                            
                            batch_idx = 0
                            while batch_idx * self.batch_size < samples_per_mod:
                                try:
                                    # Calculate indices
                                    start_idx = batch_idx * self.batch_size
                                    end_idx = min(start_idx + self.batch_size, samples_per_mod)
                                    current_batch_size = end_idx - start_idx

                                    # Generate batch data
                                    batch_data = self._generate_batch_data(
                                        current_batch_size,
                                        mod_scheme
                                    )

                                    # Validate batch data
                                    is_valid, validation_errors = self._validate_batch_data(batch_data)
                                    if not is_valid:
                                        raise ValueError(f"Batch validation failed: {validation_errors}")

                                    # Save batch data to HDF5
                                    self._save_batch_to_hdf5(
                                        f,
                                        mod_group,
                                        batch_data,
                                        start_idx,
                                        end_idx,
                                        path_loss_offset
                                    )

                                    # Update progress and stats
                                    mod_progress.update(current_batch_size)
                                    total_progress.update(current_batch_size)
                                    generation_stats['successful_batches'] += 1
                                    generation_stats['total_samples_generated'] += current_batch_size

                                    # Memory management
                                    if self.memory_callback:
                                        self.memory_callback()
                                    
                                    batch_idx += 1

                                except tf.errors.ResourceExhaustedError:
                                    self.batch_size = max(1000, self.batch_size // 2)
                                    self.logger.warning(
                                        f"Memory limit reached. Reducing batch size to {self.batch_size}"
                                    )
                                    continue
                                    
                                except Exception as e:
                                    self.logger.error(f"Batch processing error: {str(e)}")
                                    generation_stats['failed_batches'] += 1
                                    continue

                        path_loss_offset += samples_per_mod

                # Add generation metadata
                f.attrs.update({
                    'completion_time': datetime.now().isoformat(),
                    'total_samples': generation_stats['total_samples_generated'],
                    'successful_batches': generation_stats['successful_batches'],
                    'failed_batches': generation_stats['failed_batches']
                })

                # Verify dataset integrity
                if not self.validate_consistency(f):
                    raise ValueError("Dataset consistency check failed")

                # Log generation statistics
                self.logger.info("\nGeneration Statistics:")
                self.logger.info(f"Total samples generated: {generation_stats['total_samples_generated']}")
                self.logger.info(f"Successful batches: {generation_stats['successful_batches']}")
                self.logger.info(f"Failed batches: {generation_stats['failed_batches']}")

                return generation_stats['failed_batches'] == 0

        except Exception as e:
            self.logger.error(f"Dataset generation failed: {str(e)}")
            self.logger.error("Detailed error traceback:", exc_info=True)
            return False
    
    def _process_batch(self, batch_size: int, mod_scheme: str) -> Dict[str, tf.Tensor]:
        """
        Process a single batch of data with enhanced validation and debugging
        
        Args:
            batch_size: Size of the batch to process
            mod_scheme: Current modulation scheme
            
        Returns:
            Dictionary containing processed batch data
        """
        try:
            print(f"\nDEBUG - process_batch:")
            print(f"Processing batch with size {batch_size}")
            print(f"Current modulation scheme: {mod_scheme}")
            self.logger.info(f"Starting batch processing with size {batch_size} and modulation {mod_scheme}")
            
            # Validate batch size
            batch_size = tf.cast(batch_size, tf.int32)
            tf.debugging.assert_positive(batch_size, message="Batch size must be positive")
            
            # Generate QAM symbols first
            tx_symbols = self.channel_model.generate_qam_symbols(batch_size, mod_scheme)
            self.logger.debug(f"Generated TX symbols shape: {tx_symbols.shape}")
            
            # Generate SNR values
            snr_db = tf.random.uniform(
                [batch_size], 
                minval=self.system_params.snr_range[0],
                maxval=self.system_params.snr_range[1]
            )
            self.logger.debug(f"Generated SNR values shape: {snr_db.shape}")
            
            # Generate distances for path loss calculation
            distances = tf.random.uniform(
                [batch_size],
                minval=1.0,  # Minimum distance (1 meter)
                maxval=500.0,  # Maximum distance (500 meters)
                dtype=tf.float32
            )
            
            # Calculate path loss before channel generation
            path_loss = self.path_loss_manager.calculate_path_loss(
                distances,
                scenario='umi'  # or whatever scenario you're using
            )
            self.logger.debug(f"Generated path loss shape: {path_loss.shape}")

            # Ensure path loss has correct shape before channel generation
            path_loss = tf.reshape(path_loss, [batch_size, self.system_params.num_rx, self.system_params.num_tx])

            # Generate channel data with explicit shape checking
            self.logger.debug("Generating channel data...")
            channel_data = self.channel_model.generate_mimo_channel(batch_size=batch_size,snr_db=snr_db,path_loss=path_loss)          
            # Validate channel data structure
            required_keys = ['perfect_channel', 'noisy_channel']
            for key in required_keys:
                if key not in channel_data:
                    raise KeyError(f"Missing required key in channel_data: {key}")
            
            # Log channel data shapes
            self.logger.debug(f"Channel data shapes:")
            for key, tensor in channel_data.items():
                self.logger.debug(f"- {key}: {tensor.shape}")
                
            # Validate complex tensor
            from utill.tensor_shape_validator import validate_complex_tensor
            if not validate_complex_tensor(channel_data['perfect_channel']):
                raise ValueError("Invalid complex channel response")
                
            # Verify tensor shapes before metrics calculation
            expected_shapes = {
                'perfect_channel': (batch_size, self.system_params.num_rx, self.system_params.num_tx),
                'noisy_channel': (batch_size, self.system_params.num_rx, self.system_params.num_tx)
            }
            
            for key, expected_shape in expected_shapes.items():
                tf.debugging.assert_equal(
                    tf.shape(channel_data[key]),
                    expected_shape,
                    message=f"Shape mismatch for {key}"
                )
                
            # Calculate metrics with enhanced error handling
            self.logger.debug("Calculating performance metrics...")
            try:
                metrics = self.metrics_calculator.calculate_performance_metrics(
                    channel_data['perfect_channel'],
                    tx_symbols,
                    channel_data['noisy_channel'],
                    snr_db
                )
                
                # Validate metrics output
                required_metrics = ['sinr', 'spectral_efficiency', 'effective_snr', 'eigenvalues']
                for metric in required_metrics:
                    if metric not in metrics:
                        raise KeyError(f"Missing required metric: {metric}")
                        
                # Log metrics shapes
                self.logger.debug(f"Metrics shapes:")
                for key, value in metrics.items():
                    self.logger.debug(f"- {key}: {value.shape}")
                    
            except Exception as metrics_error:
                self.logger.error(f"Metrics calculation failed: {str(metrics_error)}")
                raise
                
            # Prepare and validate output
            output = {
                'channel_response': channel_data['perfect_channel'],
                'sinr': metrics['sinr'],
                'spectral_efficiency': metrics['spectral_efficiency'],
                'effective_snr': metrics['effective_snr'],
                'eigenvalues': metrics['eigenvalues']
            }
            
            # Verify batch consistency
            from utill.tensor_shape_validator import verify_batch_consistency
            if not verify_batch_consistency(output, batch_size):
                raise ValueError("Batch size inconsistency in output tensors")
                
            self.logger.info("Batch processing completed successfully")
            return output
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            self.logger.error(f"Batch size: {batch_size}, Modulation: {mod_scheme}")
            # Log additional debug information
            if 'channel_data' in locals():
                self.logger.error(f"Channel data keys: {channel_data.keys()}")
            if 'metrics' in locals():
                self.logger.error(f"Metrics keys: {metrics.keys()}")
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
        
    def verify_and_report(self, save_path: str) -> bool:
        """
        Comprehensive verification of generated dataset with detailed reporting
        
        Args:
            save_path: Path to the generated dataset
            
        Returns:
            bool: True if verification passed
        """
        try:
            # First verify using internal checks
            if not self.verify_dataset(save_path):
                self.logger.error("Internal dataset verification failed")
                return False
                
            # Then use the integrity checker for comprehensive validation
            with MIMODatasetIntegrityChecker(save_path) as checker:
                integrity_report = checker.check_dataset_integrity()
                
                if integrity_report['overall_status']:
                    self.logger.info("✅ Dataset integrity verification passed")
                    
                    # Log detailed statistics
                    if 'validation_details' in integrity_report:
                        self.logger.info("\nValidation Details:")
                        for key, stats in integrity_report['validation_details'].items():
                            self.logger.info(f"\n{key}:")
                            for stat_name, value in stats.items():
                                self.logger.info(f"  {stat_name}: {value}")
                    
                    # Log modulation scheme statistics
                    self.logger.info("\nModulation Scheme Details:")
                    for mod_scheme, details in integrity_report['modulation_schemes'].items():
                        self.logger.info(f"\n{mod_scheme}:")
                        self.logger.info(f"  Samples: {details.get('samples', 0)}")
                        self.logger.info(f"  Status: {'✅ Valid' if details.get('integrity', False) else '❌ Invalid'}")
                        
                    return True
                else:
                    self.logger.error("❌ Dataset integrity verification failed")
                    if 'errors' in integrity_report:
                        for error in integrity_report['errors']:
                            self.logger.error(f"  • {error}")
                    if 'warnings' in integrity_report:
                        for warning in integrity_report['warnings']:
                            self.logger.warning(f"  • {warning}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error during verification: {str(e)}")
            return False
            
    # In mimo_dataset_generator.py, enhance memory management:
    def _manage_memory(self):
        """
        Enhanced memory management for H100 GPUs with periodic monitoring and adaptive thresholds
        """
        try:
            # Check if monitoring interval has elapsed
            current_time = datetime.now()
            if hasattr(self, '_last_memory_check'):
                if (current_time - self._last_memory_check).seconds < 300:  # 5 minute interval
                    return
            self._last_memory_check = current_time

            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                return

            total_memory_usage = 0
            gpu_states = []

            # Monitor each GPU with enhanced metrics
            for gpu_idx, gpu in enumerate(gpus):
                try:
                    memory_info = tf.config.experimental.get_memory_info(f'GPU:{gpu_idx}')
                    current_usage = memory_info['current'] / 1e9  # GB
                    peak_usage = memory_info.get('peak', memory_info['current']) / 1e9  # GB
                    
                    # Calculate memory utilization percentage
                    memory_utilization = current_usage / self.memory_threshold if hasattr(self, 'memory_threshold') else 0.0
                    
                    gpu_states.append({
                        'index': gpu_idx,
                        'current_usage': current_usage,
                        'peak_usage': peak_usage,
                        'utilization': memory_utilization
                    })
                    
                    total_memory_usage += current_usage
                    
                    # Log detailed memory statistics
                    self.logger.info(
                        f"GPU:{gpu_idx} Memory Stats:\n"
                        f"  - Current Usage: {current_usage:.2f} GB\n"
                        f"  - Peak Usage: {peak_usage:.2f} GB\n"
                        f"  - Utilization: {memory_utilization:.2%}"
                    )

                    # Progressive memory management thresholds
                    warning_threshold = 0.75  # 75% utilization
                    critical_threshold = 0.9   # 90% utilization
                    
                    if memory_utilization > critical_threshold:
                        # Critical memory situation - aggressive reduction
                        new_batch_size = max(self.min_batch_size, self.batch_size // 2)
                        self.logger.warning(
                            f"CRITICAL: Memory utilization {memory_utilization:.2%} exceeds {critical_threshold:.2%}. "
                            f"Reducing batch size from {self.batch_size} to {new_batch_size}"
                        )
                        self.batch_size = new_batch_size
                        self.stable_iterations = 0
                        
                        # Force immediate cleanup
                        self._force_memory_cleanup()
                        
                    elif memory_utilization > warning_threshold:
                        # Warning level - moderate reduction
                        new_batch_size = max(self.min_batch_size, int(self.batch_size * 0.75))
                        self.logger.warning(
                            f"WARNING: Memory utilization {memory_utilization:.2%} exceeds {warning_threshold:.2%}. "
                            f"Reducing batch size from {self.batch_size} to {new_batch_size}"
                        )
                        self.batch_size = new_batch_size
                        self.stable_iterations = 0

                except Exception as e:
                    self.logger.warning(f"Error monitoring GPU:{gpu_idx} - {str(e)}")
                    continue

            # Update memory usage history
            if not hasattr(self, '_memory_history'):
                self._memory_history = []
            self._memory_history.append({
                'timestamp': current_time,
                'total_usage': total_memory_usage,
                'gpu_states': gpu_states
            })
            
            # Keep only last 10 memory measurements
            self._memory_history = self._memory_history[-10:]

            # Check if memory usage is stable and try to grow batch size
            if self._is_memory_stable():
                self.safe_memory_growth()

        except Exception as e:
            self.logger.error(f"Memory management failed: {str(e)}")
            self.batch_size = self.min_batch_size
            self._force_memory_cleanup()

    def _force_memory_cleanup(self):
        """Force cleanup of memory resources"""
        try:
            # Clear TensorFlow session
            tf.keras.backend.clear_session()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear any cached tensors
            tf.debugging.set_log_device_placement(False)
            
            # Optional: Clear CUDA cache if available
            try:
                torch.cuda.empty_cache()  # If PyTorch is available
            except:
                pass
                
            self.logger.info("Forced memory cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {str(e)}")

    def _is_memory_stable(self) -> bool:
        """Check if memory usage has been stable for recent history"""
        if len(self._memory_history) < 5:  # Need at least 5 measurements
            return False
            
        recent_usage = [state['total_usage'] for state in self._memory_history[-5:]]
        mean_usage = sum(recent_usage) / len(recent_usage)
        max_deviation = max(abs(usage - mean_usage) for usage in recent_usage)
        
        # Consider stable if max deviation is less than 5% of mean
        return max_deviation < (mean_usage * 0.05)


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

    def _check_batch_size(self, requested_batch_size: int) -> int:
        """
        Validate and adjust batch size based on hardware capabilities
        """
        try:
            # Get max batch size from path loss manager
            max_allowed = self.path_loss_manager.max_batch_size
            
            if requested_batch_size > max_allowed:
                self.logger.warning(
                    f"Batch size {requested_batch_size} exceeds max allowed ({max_allowed}). "
                    f"Reducing to {max_allowed}."
                )
                return max_allowed
            
            return requested_batch_size
            
        except Exception as e:
            self.logger.warning(f"Error checking batch size: {str(e)}")
            return 4000  # Safe default for H100
    
    def _check_batch_size_safety(self, batch_size: int) -> int:
        """
        Ensure batch size is compatible with tensor dimensions and memory constraints
        
        Args:
            batch_size (int): Initial batch size to check
            
        Returns:
            int: Safe and adjusted batch size
        """
        try:
            # Print initial batch size for debugging
            print(f"\nInitial batch size before safety checks: {batch_size}")
            
            # Get max_batch_size from PathLossManager
            max_batch_size = getattr(self.path_loss_manager.system_params, 'max_batch_size', self.max_batch_size)
            print(f"Maximum allowed batch size from PathLossManager: {max_batch_size}")
            
            # Ensure integer type for batch size
            batch_size = int(tf.cast(batch_size, tf.int32).numpy())
            
            # Check against maximum allowed batch size
            if batch_size > max_batch_size:
                self.logger.warning(
                    f"Batch size {batch_size} exceeds max allowed by PathLossManager ({max_batch_size}). "
                    f"Reducing to {max_batch_size}."
                )
                batch_size = int(max_batch_size)
            
            # Get MIMO dimensions
            num_rx = self.system_params.num_rx
            num_tx = self.system_params.num_tx
            print(f"MIMO dimensions - RX: {num_rx}, TX: {num_tx}")
            
            # Calculate total elements
            total_elements = batch_size * num_rx * num_tx
            print(f"Total elements with current batch size: {total_elements}")
            
            # Calculate maximum allowed elements
            max_elements = 9000 * num_rx * num_tx  # 9000 is the safe maximum batch size
            print(f"Maximum allowed elements: {max_elements}")
            
            # Check if total elements exceeds maximum
            if total_elements > max_elements:
                adjusted_batch_size = max_elements // (num_rx * num_tx)
                self.logger.warning(
                    f"Total elements {total_elements} exceeds maximum {max_elements}. "
                    f"Adjusting batch size from {batch_size} to {adjusted_batch_size}"
                )
                batch_size = adjusted_batch_size
                total_elements = batch_size * num_rx * num_tx
            
            # Verify tensor dimensions compatibility
            if total_elements % (num_rx * num_tx) != 0:
                adjusted_batch_size = total_elements // (num_rx * num_tx)
                self.logger.warning(
                    f"Batch size {batch_size} not compatible with MIMO dimensions ({num_rx}x{num_tx}). "
                    f"Adjusting to {adjusted_batch_size}"
                )
                batch_size = adjusted_batch_size
            
            # Apply minimum batch size constraint
            min_batch_size = 1000  # Minimum safe batch size
            if batch_size < min_batch_size:
                self.logger.warning(
                    f"Batch size {batch_size} below minimum {min_batch_size}. "
                    f"Setting to minimum."
                )
                batch_size = min_batch_size
            
            # Final validation
            final_elements = batch_size * num_rx * num_tx
            print(f"\nFinal validation:")
            print(f"- Final batch size: {batch_size}")
            print(f"- Final total elements: {final_elements}")
            print(f"- Dimensions check: {final_elements % (num_rx * num_tx) == 0}")
            print(f"- Maximum check: {final_elements <= max_elements}")
            
            self.logger.info(f"Final adjusted batch size: {batch_size}")
            return batch_size
            
        except Exception as e:
            error_msg = f"Error in batch size safety check: {str(e)}"
            print(f"\nERROR: {error_msg}")
            self.logger.error(error_msg)
            return 1000  # Conservative fallback

    def generate_batch_data(self, batch_idx: int, batch_size: int):
        """
        Generate batch data with proper integer types for the current batch
        
        Args:
            batch_idx (int): Index of the current batch
            batch_size (int): Size of the batch to generate
            
        Returns:
            tuple: Generated distances and SNR values for the batch
        """
        try:
            # Ensure batch_size is int32
            batch_size = tf.cast(batch_size, tf.int32)
            
            # Generate distances
            distances = tf.random.uniform(
                shape=[batch_size],
                minval=self.system_params.min_distance,
                maxval=self.system_params.max_distance,
                dtype=tf.float32
            )
            
            # Generate SNR values
            snr_db = tf.random.uniform(
                shape=[batch_size],
                minval=self.system_params.snr_range[0],
                maxval=self.system_params.snr_range[1],
                dtype=tf.float32
            )
            
            self.logger.debug(f"Generated batch {batch_idx} with size {batch_size}")
            return distances, snr_db
            
        except Exception as e:
            self.logger.error(f"Error generating batch data: {str(e)}")
            raise

    def _generate_channel_samples(self, batch_size: int):
        """
        Generate MIMO channel samples with proper integer types
        
        Args:
            batch_size (int): Size of the batch to generate
            
        Returns:
            tf.Tensor: Generated complex channel matrix
        """
        try:
            # Ensure batch_size is int32
            batch_size = tf.cast(batch_size, tf.int32)
            
            # Define shape as integer tensor
            shape = tf.convert_to_tensor(
                [batch_size, self.system_params.num_rx, self.system_params.num_tx], 
                dtype=tf.int32
            )
            
            # Generate complex channel matrix
            channel_matrix = tf.complex(
                tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32),
                tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32)
            ) / tf.sqrt(2.0)
            
            self.logger.debug(
                f"Generated channel samples with shape: {channel_matrix.shape}"
            )
            return channel_matrix
            
        except Exception as e:
            self.logger.error(f"Error generating channel samples: {str(e)}")
            raise

    def _calculate_tensor_dimensions(self, batch_size: int) -> tuple:
        """
        Calculate correct tensor dimensions based on batch size and system parameters
        
        Args:
            batch_size (int): Current batch size
            
        Returns:
            tuple: Expected tensor dimensions
        """
        num_rx = self.system_params.num_rx
        num_tx = self.system_params.num_tx
        
        # Calculate total elements
        total_elements = batch_size * num_rx * num_tx
        
        # Return expected dimensions
        return (batch_size, num_rx, num_tx)

# Example usage
def main():
    generator = MIMODatasetGenerator()
    generator.generate_dataset(num_samples=1_320_000)
    generator.verify_dataset('dataset/mimo_dataset.h5')

if __name__ == "__main__":
    main()