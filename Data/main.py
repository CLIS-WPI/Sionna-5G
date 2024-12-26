# main.py
# MIMO Dataset Generation Command-Line Interface
# Provides flexible command-line configuration for dataset generation and verification
# Manages system configuration, logging, and dataset generation workflow

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

# Add at the top of your main.py after imports

# Set extremely conservative CUDA memory settings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TF_MEMORY_ALLOCATION'] = '0.3'  # Only use 30% of memory
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.3'

def configure_device():
    """Configure and detect available processing device"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    # Enable memory growth
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Set very conservative memory limit (30% of available memory)
                    memory_limit = int(11*1024*0.3)  # Set to 30% of memory in MB
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                    )
                
                print(f"Found {len(gpus)} GPU(s). Using GPU for processing.")
                
                if len(gpus) > 1:
                    # Use more conservative strategy for multiple GPUs
                    options = tf.distribute.MirroredStrategyOptions(
                        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
                    )
                    strategy = tf.distribute.MirroredStrategy(options=options)
                    print(f"Using {len(gpus)} GPUs with MirroredStrategy")
                    return strategy
                return True
                
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
                print("Falling back to CPU")
                return False
        else:
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
        default=20_000_000,#900_000, 
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
    Main entry point for MIMO dataset generation
    """
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Configure logging with detailed format
        logger = configure_logging(
            log_level=args.log_level,
            log_file=f'logs/mimo_dataset_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]'
        )
        
        # Configure system parameters
        system_params = configure_system_parameters(args)
        
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

        # Configure device before creating generator
        device_config = configure_device()
        logger.info("Device configuration completed")
        if isinstance(device_config, tf.distribute.Strategy):
            logger.info("Using distributed strategy with multiple GPUs")
        elif device_config:
            logger.info("Using single GPU")
        else:
            logger.info("Using CPU")
        
        # Add GPU memory info if available
        if tf.config.list_physical_devices('GPU'):
            try:
                memory_info = tf.config.experimental.get_memory_info('GPU:0')
                logger.info(f"GPU memory usage: {memory_info['current'] / 1e9:.2f} GB")
            except:
                logger.info("GPU memory information not available")

        # Create and configure dataset generator
        logger.debug("Initializing dataset generator...")
        generator = MIMODatasetGenerator(
            system_params=system_params,
            logger=logger
        )
        
        # Generate dataset
        logger.info("Starting dataset generation...")
        try:
            generator.generate_dataset(
                num_samples=system_params.total_samples,
                save_path=output_path
            )
            logger.info("Dataset generation completed")
            
            # Check if the file exists and has content
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"Generated file size: {file_size} bytes")
                if file_size == 0:
                    raise ValueError("Generated dataset file is empty")
            else:
                raise FileNotFoundError("Dataset file was not created")
        except Exception as e:
            logger.error(f"Dataset generation failed: {str(e)}", exc_info=True)
            return 1

        # Verification section
        if args.verify:
            logger.info("Starting dataset verification...")
            try:
                # Check if file exists and is not empty
                if not os.path.exists(output_path):
                    logger.error("Dataset file does not exist")
                    return 1
                    
                file_size = os.path.getsize(output_path)
                if file_size == 0:
                    logger.error("Dataset file is empty")
                    return 1
                    
                logger.info(f"Dataset file size: {file_size} bytes")
                
                with MIMODatasetIntegrityChecker(output_path) as checker:
                    # Log dataset structure
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
        logger.error(f"Critical error during execution: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
