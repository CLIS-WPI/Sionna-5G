# dataset_generator/mimo_dataset_generator.py
# Comprehensive MIMO Dataset Generation Framework
# Generates large-scale, configurable MIMO communication datasets with multiple modulation schemes
# Supports advanced channel modeling, metrics calculation, and dataset verification

import os
import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from config.system_parameters import SystemParameters
from core.channel_model import ChannelModelManager
from core.metrics_calculator import MetricsCalculator
from core.path_loss_model import PathLossManager
from utill.logging_config import LoggerManager
from utill.tensor_shape_validator import validate_tensor_shapes  
from datetime import datetime
from integrity.dataset_integrity_checker import MIMODatasetIntegrityChecker

class MIMODatasetGenerator:
    """
    Comprehensive MIMO dataset generation framework
    """
    
    def __init__(
        self, 
        system_params: SystemParameters = None,
        logger=None
    ):
        """
        Initialize MIMO dataset generator
        
        Args:
            system_params (SystemParameters, optional): System configuration
            logger (logging.Logger, optional): Logger instance
        """
        # Use default system parameters if not provided
        self.system_params = system_params or SystemParameters()
        
        # Configure logger
        self.logger = logger or LoggerManager.get_logger(
            name='MIMODatasetGenerator', 
            log_level='INFO'
        )
        
        # Initialize core components
        self.channel_model = ChannelModelManager(self.system_params)
        self.metrics_calculator = MetricsCalculator(self.system_params)
        self.path_loss_manager = PathLossManager(self.system_params)
        self.integrity_checker = None
        
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
                    if total_memory > 8.0:  # More than 8GB available
                        self.batch_size = 64000
                    elif total_memory > 4.0:  # More than 4GB available
                        self.batch_size = 32000
                    else:  # Limited memory
                        self.batch_size = 19000
                        
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
                
                self.memory_callback = self.memory_monitoring_callback
                
            
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.logger.info(f"Prepared output directory: {os.path.dirname(save_path)}")
    
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
            
            path_loss_data = hdf5_file.create_group('path_loss_data')
            path_loss_data.attrs['description'] = 'Path loss measurements and calculations'
            
            config_data = hdf5_file.create_group('configuration')
            config_data.attrs['description'] = 'System configuration parameters'
            
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
                mod_group = modulation_data.create_group(mod_scheme)
                mod_group.attrs['description'] = f'Data for {mod_scheme} modulation'
                
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

            # Create path loss datasets
            for name in ['fspl', 'scenario_pathloss']:
                dataset = path_loss_data.create_dataset(
                    name,
                    shape=(num_samples,),
                    dtype=np.float32,
                    chunks=True,
                    compression='gzip'
                )
                dataset.attrs['description'] = f'{name.upper()} measurements'
                dataset.attrs['units'] = 'dB'

            self.logger.info("Dataset structure created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create dataset structure: {str(e)}")
            raise

    def generate_dataset(
        self, 
        num_samples: int, 
        save_path: str = 'dataset/mimo_dataset.h5'
    ):
        """
        Generate comprehensive MIMO dataset with enhanced shape validation and memory management
        """
        try:
            self._prepare_output_directory(save_path)

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
            
            with h5py.File(save_path, 'w') as f:
                self._create_dataset_structure(f, num_samples)
                
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
                            distances = tf.random.uniform([self.batch_size], 10.0, 500.0)
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
                                self.logger.error(f"Failed to generate valid batch after {MAX_RETRIES} attempts")
                                continue

                            # Calculate path loss data
                            fspl = self.path_loss_manager.calculate_free_space_path_loss(distances)
                            scenario_pl = self.path_loss_manager.calculate_path_loss(distances, 'umi')
                            
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
                            
                            # Save path loss data
                            f['path_loss_data']['fspl'][start_idx:end_idx] = fspl.numpy()
                            f['path_loss_data']['scenario_pathloss'][start_idx:end_idx] = scenario_pl.numpy()
                            
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
                    
                    mod_progress.close()
                
                total_progress.close()

            # Verify the complete dataset
            if self.verify_dataset(save_path):
                self.logger.info(f"Dataset successfully generated and verified at {save_path}")
            else:
                self.logger.warning("Dataset generated but failed verification")
            
        except Exception as e:
            self.logger.error(f"Dataset generation failed: {str(e)}")
            self.logger.error("Detailed error traceback:", exc_info=True)
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

    
    def verify_dataset(self, save_path: str) -> bool:
        """
        Verify dataset integrity using MIMODatasetIntegrityChecker
        """
        try:
            self.integrity_checker = MIMODatasetIntegrityChecker(save_path)
            integrity_report = self.integrity_checker.check_dataset_integrity()
            
            if integrity_report['overall_status']:
                self.logger.info("Dataset verification successful")
                return True
            else:
                self.logger.warning("Dataset verification failed")
                # Add detailed error logging
                if 'errors' in integrity_report:
                    for error in integrity_report['errors']:
                        self.logger.warning(f"Verification error: {error}")
                
                # Log statistics for debugging
                if 'modulation_schemes' in integrity_report:
                    for mod_scheme, stats in integrity_report['modulation_schemes'].items():
                        self.logger.warning(f"\nModulation scheme {mod_scheme}:")
                        self.logger.warning(f"Samples: {stats.get('samples', 'N/A')}")
                        self.logger.warning(f"Status: {stats.get('integrity', False)}")
                        
                        if 'datasets' in stats:
                            for dataset_name, dataset_info in stats['datasets'].items():
                                self.logger.warning(f"\n{dataset_name}:")
                                self.logger.warning(f"Shape: {dataset_info.get('shape', 'N/A')}")
                                self.logger.warning(f"Statistics: {dataset_info.get('statistics', 'N/A')}")
                
                return False
                
        except Exception as e:
            self.logger.error(f"Dataset verification error: {str(e)}")
            return False
        
    def _manage_memory(self):
        """Memory management helper"""
        if tf.config.list_physical_devices('GPU'):
            tf.keras.backend.clear_session()
        import gc
        gc.collect()

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
        Check and adjust batch size based on available memory
        """
        try:
            import psutil
            available_memory = psutil.virtual_memory().available
            # Calculate approximate memory requirement per sample
            sample_size = (self.system_params.num_tx * 
                        self.system_params.num_rx * 
                        8 * 4)  # 8 bytes per complex64, factor of 4 for safety
            max_safe_batch = int(available_memory * 0.5 / sample_size)  # Use 50% of available memory
            return min(batch_size, max_safe_batch)
        except:
            return min(batch_size, 5000)  # Conservative default
        
# Example usage
def main():
    generator = MIMODatasetGenerator()
    generator.generate_dataset(num_samples=12_000_000)
    generator.verify_dataset('dataset/mimo_dataset.h5')

if __name__ == "__main__":
    main()