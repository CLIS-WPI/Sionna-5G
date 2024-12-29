# main.py
# MIMO Dataset Generation Command-Line Interface
# Provides flexible command-line configuration for dataset generation and verification
# Manages system configuration, logging, and dataset generation workflow
import os
import sys
import argparse
import logging
from datetime import datetime
import os
from config.system_parameters import SystemParameters
from dataset_generator.mimo_dataset_generator import MIMODatasetGenerator
from utill.logging_config import configure_logging, LoggerManager
from integrity.dataset_integrity_checker import MIMODatasetIntegrityChecker
import h5py
import tensorflow as tf
from typing import Dict
import tensorflow as tf
tf.config.set_soft_device_placement(True)
# Set conservative CUDA memory settings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU

def clear_gpu_memory():
    tf.keras.backend.clear_session()
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.reset_memory_stats(gpu)
        except:
            pass

def configure_device():
    """
    Configure GPU devices with conservative memory settings
    """
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Set memory growth first, before any other configurations
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Remove the logical device configuration since it conflicts
                # with memory growth setting
                print(f"Found {len(gpus)} GPU(s). Using GPU for processing.")
                return True
                
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
                return False
                
        print("No GPU found. Using CPU for processing")
        return False
        
    except Exception as e:
        print(f"Error during device configuration: {e}")
        return False
    
def parse_arguments():
    """
    Parse command-line arguments for dataset generation
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="MIMO Dataset Generation Framework"
    )
    
    parser.add_argument(
        '--samples', 
        type=int, 
        default=21_000_000,#900_000, 
        help="Number of samples to generate (default: 1,000,000)"
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help="Batch size for processing (default: auto-configured based on available memory)"
    )
    
        
    parser.add_argument(
        '--output', 
        type=str, 
        default=None, 
        help="Output path for dataset (default: dataset/mimo_dataset_TIMESTAMP.h5)"
    )
    
    parser.add_argument(
        '--log-level', 
        type=str, 
        default='INFO', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        '--verify', 
        action='store_true', 
        help="Verify generated dataset after creation"
    )
    
    parser.add_argument(
        '--tx-antennas', 
        type=int, 
        default=4, 
        help="Number of transmit antennas (default: 4)"
    )
    
    parser.add_argument(
        '--rx-antennas', 
        type=int, 
        default=4, 
        help="Number of receive antennas (default: 4)"
    )
    
    return parser.parse_args()

def configure_system_parameters(args):
    """
    Configure system parameters based on command-line arguments
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments
    
    Returns:
        SystemParameters: Configured system parameters
    """
    return SystemParameters(
        num_tx=args.tx_antennas,
        num_rx=args.rx_antennas,
        total_samples=args.samples
    )

def generate_output_path(base_path=None):
    """
    Generate a unique output path for the dataset
    
    Args:
        base_path (str, optional): Base path for dataset
    
    Returns:
        str: Unique output path
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base_path:
            return base_path
            
    counter = 0
    while True:
            suffix = f"_{counter}" if counter > 0 else ""
            path = f"dataset/mimo_dataset_{timestamp}{suffix}.h5"
            if not os.path.exists(path):
                return path
            counter += 1

def main():
    """
    Main entry point for MIMO dataset generation with enhanced GPU and memory management
    """
    try:
        clear_gpu_memory()
        # Parse arguments first
        args = parse_arguments()
        
        # Initialize logger before any other operations
        logger = configure_logging(
            log_level=args.log_level,
            log_file=f'logs/mimo_dataset_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]'
        )

        # Then configure device
        device_config = configure_device()
        if not device_config:
            logger.warning("GPU configuration failed, falling back to CPU")
            
        # Clear GPU memory
        logger.info("Clearing GPU memory and configuring device...")
        tf.keras.backend.clear_session()
        
        # Configure system parameters
        system_params = configure_system_parameters(args)
        system_params.replay_buffer_size = min(system_params.replay_buffer_size, 100000)
        
        # Set conservative batch size if not specified
        if args.batch_size is None:
            args.batch_size = 1000  # Start with a very conservative batch size
        
        # Generate output path
        output_path = generate_output_path(args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Log configuration details
        logger.info("=== Configuration Details ===")
        logger.info(f"Total samples: {system_params.total_samples}")
        logger.info(f"TX Antennas: {system_params.num_tx}")
        logger.info(f"RX Antennas: {system_params.num_rx}")
        logger.info(f"Output path: {output_path}")
        logger.info(f"Verification enabled: {args.verify}")
        logger.info(f"Batch size: {args.batch_size}")

        # Configure device with enhanced error handling
        device_config = configure_device()
        logger.info("Device configuration completed")
        
        if isinstance(device_config, tf.distribute.Strategy):
            logger.info("Using distributed strategy with multiple GPUs")
        elif device_config:
            logger.info("Using single GPU")
        else:
            logger.info("Using CPU")
        
        # Monitor GPU memory with enhanced error handling
        if tf.config.list_physical_devices('GPU'):
            try:
                memory_info = tf.config.experimental.get_memory_info('GPU:0')
                logger.info(f"Initial GPU memory usage: {memory_info['current'] / 1e9:.2f} GB")
                
                # Add memory monitoring callback
                def memory_monitoring_callback():
                    try:
                        current_memory = tf.config.experimental.get_memory_info('GPU:0')['current'] / 1e9
                        if current_memory > 0.8:  # If using more than 80% memory
                            logger.warning(f"High GPU memory usage detected: {current_memory:.2f} GB")
                    except:
                        pass
                
            except Exception as e:
                logger.warning(f"GPU memory monitoring not available: {e}")

        # Create and configure dataset generator with memory cleanup
        logger.debug("Initializing dataset generator...")
        import gc
        gc.collect()  # Force garbage collection before creating generator
        
        generator = MIMODatasetGenerator(
            system_params=system_params,
            logger=logger
        )
        
        # Generate dataset with enhanced error handling
        logger.info("Starting dataset generation...")
        try:
            generator.generate_dataset(
                num_samples=system_params.total_samples,
                save_path=output_path
            )
            logger.info("Dataset generation completed")
            
            # Verify output file
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"Generated file size: {file_size / (1024*1024):.2f} MB")
                if file_size == 0:
                    raise ValueError("Generated dataset file is empty")
            else:
                raise FileNotFoundError("Dataset file was not created")
                
        except tf.errors.ResourceExhaustedError:
            logger.error("GPU memory exhausted. Try reducing batch size or total samples.")
            return 1
        except Exception as e:
            logger.error(f"Dataset generation failed: {str(e)}", exc_info=True)
            return 1

        # Verification section with enhanced memory management
        if args.verify:
            logger.info("Starting dataset verification...")
            try:
                # Clear memory before verification
                tf.keras.backend.clear_session()
                gc.collect()
                
                if not os.path.exists(output_path):
                    logger.error("Dataset file does not exist")
                    return 1
                    
                file_size = os.path.getsize(output_path)
                if file_size == 0:
                    logger.error("Dataset file is empty")
                    return 1
                    
                logger.info(f"Dataset file size: {file_size / (1024*1024):.2f} MB")
                
                with MIMODatasetIntegrityChecker(output_path) as checker:
                    # Log dataset structure with enhanced error handling
                    try:
                        with h5py.File(output_path, 'r') as f:
                            logger.debug("=== Dataset Structure ===")
                            logger.debug(f"Root groups: {list(f.keys())}")
                            
                            if 'modulation_data' in f:
                                mod_schemes = list(f['modulation_data'].keys())
                                logger.debug(f"Modulation schemes: {mod_schemes}")
                                
                                for scheme in mod_schemes:
                                    scheme_group = f['modulation_data'][scheme]
                                    logger.debug(f"\nModulation scheme: {scheme}")
                                    logger.debug(f"Available datasets: {list(scheme_group.keys())}")
                                    
                                    for dataset_name in ['channel_response', 'sinr', 'spectral_efficiency']:
                                        if dataset_name in scheme_group:
                                            shape = scheme_group[dataset_name].shape
                                            logger.debug(f"  {dataset_name} shape: {shape}")
                    except Exception as e:
                        logger.error(f"Error examining dataset structure: {str(e)}")
                        return 1
                    
                    # Perform integrity checks
                    logger.info("\nPerforming integrity checks...")
                    integrity_report = checker.check_dataset_integrity()
                    
                    if integrity_report.get('overall_status', False):
                        logger.info("✅ Dataset verification successful")
                    else:
                        logger.warning("❌ Dataset verification failed")
                        if 'errors' in integrity_report:
                            for error in integrity_report['errors']:
                                logger.error(f"  • {error}")
                        return 1
            except Exception as e:
                logger.error(f"Verification process failed: {str(e)}")
                return 1
        
        return 0

    except Exception as e:
        # Create a basic logger if the main logger hasn't been initialized
        if 'logger' not in locals():
            logger = logging.getLogger()
            logger.setLevel(logging.ERROR)
            handler = logging.StreamHandler()
            logger.addHandler(handler)
        
        logger.error(f"Critical error during execution: {str(e)}", exc_info=True)
        return 1
    
if __name__ == "__main__":
    sys.exit(main())
