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
from datetime import datetime

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
    
    def _prepare_output_directory(self, save_path: str):
        """
        Prepare output directory for dataset
        
        Args:
            save_path (str): Path to save dataset
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.logger.info(f"Prepared output directory: {os.path.dirname(save_path)}")
    
    def _create_dataset_structure(self, hdf5_file, num_samples: int):
        """
        Create HDF5 dataset structure with comprehensive validation and documentation
        
        Args:
            hdf5_file: HDF5 file object to create datasets in
            num_samples (int): Total number of samples to generate
            
        Raises:
            ValueError: If num_samples is not positive or not divisible by number of modulation schemes
            RuntimeError: If HDF5 group creation fails
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
                
                # Channel and metrics datasets with proper dtypes
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
                        chunks=True,  # Enable chunking for better I/O performance
                        compression='gzip',  # Enable compression
                        compression_opts=4  # Compression level
                    )
                    # Add description
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
        Generate comprehensive MIMO dataset
        
        Args:
            num_samples (int): Total number of samples to generate
            save_path (str): Path to save HDF5 dataset
        """
        try:
            # Prepare output directory
            self._prepare_output_directory(save_path)
            
            # Batch size and samples per modulation
            batch_size = min(4096, num_samples)
            samples_per_mod = num_samples // len(self.system_params.modulation_schemes)
            
            # Create HDF5 file
            with h5py.File(save_path, 'w') as f:
                # Create dataset structure
                self._create_dataset_structure(f, num_samples)
                
                # Progress tracking
                total_progress = tqdm(
                    total=num_samples, 
                    desc="Total Dataset Generation", 
                    unit="samples"
                )
                
                # Generate data for each modulation scheme
                for mod_scheme in self.system_params.modulation_schemes:
                    mod_group = f['modulation_data'][mod_scheme]
                    
                    mod_progress = tqdm(
                        total=samples_per_mod, 
                        desc=f"{mod_scheme} Generation", 
                        unit="samples", 
                        leave=False
                    )
                    
                    # Iterate through batches
                    for batch_idx in range(samples_per_mod // batch_size):
                        start_idx = batch_idx * batch_size
                        end_idx = start_idx + batch_size
                        
                        # Generate random distances and SNR
                        distances = tf.random.uniform([batch_size], 10.0, 500.0)
                        snr_db = tf.random.uniform(
                            [batch_size], 
                            self.system_params.snr_range[0], 
                            self.system_params.snr_range[1]
                        )
                        
                        # Generate channel samples and symbols
                        h_perfect, h_noisy = self.channel_model.generate_channel_samples(batch_size, snr_db)
                        
                        # Fix channel response shape
                        if len(h_perfect.shape) == 4:  # If shape is (batch, batch, rx, tx)
                            h_perfect = h_perfect[:, 0, :, :]  # Take first slice to get (batch, rx, tx)
                        
                        # Apply path loss to properly shaped channel response
                        h_with_pl = self.path_loss_manager.apply_path_loss(h_perfect, distances)
                        
                        # Generate and process symbols
                        tx_symbols = self.channel_model.generate_qam_symbols(batch_size, mod_scheme)
                        tx_symbols = tf.expand_dims(tx_symbols, -1)  # Add dimension for matrix multiplication
                        rx_symbols = tf.matmul(h_with_pl, tx_symbols)
                        
                        # Calculate metrics with correct tensor shapes
                        metrics = self.metrics_calculator.calculate_performance_metrics(
                            h_with_pl,  # Now should be shape (batch_size, num_rx, num_tx)
                            tx_symbols,
                            rx_symbols,
                            snr_db
                        )

                        # Process metrics to ensure correct shapes
                        effective_snr = tf.squeeze(metrics['effective_snr'])
                        spectral_efficiency = tf.squeeze(metrics['spectral_efficiency'])
                        sinr = tf.squeeze(metrics['sinr'])
                        eigenvalues = metrics['eigenvalues']
                        
                        # Ensure all metrics have correct shape
                        if len(effective_snr.shape) > 1:
                            effective_snr = tf.reduce_mean(effective_snr, axis=1)
                        if len(spectral_efficiency.shape) > 1:
                            spectral_efficiency = tf.reduce_mean(spectral_efficiency, axis=1)
                        if len(sinr.shape) > 1:
                            sinr = tf.reduce_mean(sinr, axis=1)

                        # Save processed data to HDF5
                        mod_group['channel_response'][start_idx:end_idx] = h_with_pl.numpy()
                        mod_group['sinr'][start_idx:end_idx] = sinr.numpy()
                        mod_group['spectral_efficiency'][start_idx:end_idx] = spectral_efficiency.numpy()
                        mod_group['effective_snr'][start_idx:end_idx] = effective_snr.numpy()
                        mod_group['eigenvalues'][start_idx:end_idx] = eigenvalues.numpy()
                        
                        # Calculate enhanced metrics
                        enhanced_metrics = self.metrics_calculator.calculate_enhanced_metrics(
                            h_with_pl, tx_symbols, rx_symbols, snr_db
                        )
                        
                        mod_group['ber'][start_idx:end_idx] = enhanced_metrics['ber']
                        mod_group['throughput'][start_idx:end_idx] = enhanced_metrics['throughput']
                        
                        # Save path loss data
                        f['path_loss_data']['fspl'][start_idx:end_idx] = (
                            self.path_loss_manager.calculate_free_space_path_loss(distances).numpy()
                        )
                        f['path_loss_data']['scenario_pathloss'][start_idx:end_idx] = (
                            self.path_loss_manager.calculate_path_loss(distances).numpy()
                        )
                        
                        # Update progress
                        mod_progress.update(batch_size)
                        total_progress.update(batch_size)
                    
                    mod_progress.close()
                
                total_progress.close()
            
            self.logger.info(f"Dataset successfully generated at {save_path}")
        
        except Exception as e:
            self.logger.error(f"Dataset generation failed: {str(e)}")
            self.logger.error(f"Detailed error traceback:", exc_info=True)
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

    
    def verify_dataset(self, save_path: str):
        """
        Verify dataset integrity and generate statistics
        
        Args:
            save_path (str): Path to HDF5 dataset
        """
        try:
            with h5py.File(save_path, 'r') as f:
                # Detailed dataset verification logic
                total_size = 0
                
                for mod_scheme in self.system_params.modulation_schemes:
                    mod_group = f['modulation_data'][mod_scheme]
                    
                    self.logger.info(f"\n{mod_scheme} Dataset Statistics:")
                    for dataset_name, dataset in mod_group.items():
                        size_mb = dataset.size * dataset.dtype.itemsize / (1024*1024)
                        total_size += size_mb
                        
                        self.logger.info(f"  {dataset_name}:")
                        self.logger.info(f"    Shape: {dataset.shape}")
                        self.logger.info(f"    Type: {dataset.dtype}")
                        self.logger.info(f"    Size: {size_mb:.2f} MB")
                
                self.logger.info(f"\nTotal dataset size: {total_size:.2f} MB")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Dataset verification failed: {e}")
            return False

# Example usage
def main():
    generator = MIMODatasetGenerator()
    generator.generate_dataset(num_samples=100_000)
    generator.verify_dataset('dataset/mimo_dataset.h5')

if __name__ == "__main__":
    main()