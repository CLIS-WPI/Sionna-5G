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
    
    def _create_dataset_structure(
        self, 
        hdf5_file, 
        num_samples: int
    ):
        """
        Create HDF5 dataset structure
        
        Args:
            hdf5_file (h5py.File): HDF5 file object
            num_samples (int): Total number of samples
        """
        # Samples per modulation scheme
        samples_per_mod = num_samples // len(self.system_params.modulation_schemes)
        
        # Create groups
        modulation_data = hdf5_file.create_group('modulation_data')
        path_loss_data = hdf5_file.create_group('path_loss_data')
        config_data = hdf5_file.create_group('configuration')
        
        # Create datasets for each modulation scheme
        for mod_scheme in self.system_params.modulation_schemes:
            mod_group = modulation_data.create_group(mod_scheme)
            
            # Channel and metrics datasets
            datasets = {
                'channel_response': (samples_per_mod, self.system_params.num_rx, self.system_params.num_tx),
                'sinr': (samples_per_mod,),
                'spectral_efficiency': (samples_per_mod,),
                'effective_snr': (samples_per_mod,),
                'eigenvalues': (samples_per_mod, self.system_params.num_rx),
                'ber': (samples_per_mod,),
                'throughput': (samples_per_mod,),
                'inference_time': (samples_per_mod,)
            }
            
            for name, shape in datasets.items():
                mod_group.create_dataset(
                    name, 
                    shape=shape, 
                    dtype=np.float32 if name != 'channel_response' else np.complex64
                )
        
        # Path loss datasets
        path_loss_data.create_dataset('fspl', shape=(num_samples,), dtype=np.float32)
        path_loss_data.create_dataset('scenario_pathloss', shape=(num_samples,), dtype=np.float32)
        
        # Store configuration parameters
        for key, value in self.system_params.get_config_dict().items():
            config_data.attrs[key] = str(value)
    
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
                        
                        # Generate random distances
                        distances = tf.random.uniform(
                            [batch_size], 10.0, 500.0
                        )
                        
                        # Generate random SNR
                        snr_db = tf.random.uniform(
                            [batch_size], 
                            self.system_params.snr_range[0], 
                            self.system_params.snr_range[1]
                        )
                        
                        # Generate channel samples
                        h_perfect, h_noisy = self.channel_model.generate_channel_samples(
                            batch_size, snr_db
                        )
                        
                        # Apply path loss
                        h_with_pl = self.path_loss_manager.apply_path_loss(
                            h_perfect, distances
                        )
                        
                        # Generate symbols
                        tx_symbols = self.channel_model.generate_qam_symbols(batch_size, mod_scheme)
                        
                        # Simulate transmission
                        rx_symbols = tf.matmul(h_with_pl, tx_symbols)
                        
                        # Calculate metrics
                        metrics = self.metrics_calculator.calculate_enhanced_metrics(
                            h_with_pl, tx_symbols, rx_symbols, snr_db
                        )
                        
                        # Save data to HDF5
                        mod_group['channel_response'][start_idx:end_idx] = h_with_pl.numpy()
                        mod_group['sinr'][start_idx:end_idx] = metrics['sinr'].numpy()
                        mod_group['spectral_efficiency'][start_idx:end_idx] = metrics['spectral_efficiency'].numpy()
                        mod_group['effective_snr'][start_idx:end_idx] = metrics['effective_snr'].numpy()
                        mod_group['eigenvalues'][start_idx:end_idx] = metrics['eigenvalues'].numpy()
                        mod_group['ber'][start_idx:end_idx] = metrics['ber']
                        mod_group['throughput'][start_idx:end_idx] = metrics['throughput']
                        
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
            self.logger.error(f"Dataset generation failed: {e}")
            raise
    
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