# main.py
# MIMO Dataset Generation Command-Line Interface
# Provides flexible command-line configuration for dataset generation and verification
# Manages system configuration, logging, and dataset generation workflow
import os
import sys
import argparse
import logging
from datetime import datetime
from config.system_parameters import SystemParameters
from dataset_generator.mimo_dataset_generator import MIMODatasetGenerator
from utill.logging_config import configure_logging, LoggerManager
from integrity.dataset_integrity_checker import MIMODatasetIntegrityChecker
import h5py
import tensorflow as tf
from typing import Dict
import gc

def configure_gpu_environment():
    """
    Combined function to clean GPU memory and configure devices with optimized settings
    
    Returns:
        Tuple[bool, Dict]: (Success status, GPU configuration details)
    """
    try:
        # Clear existing sessions and cache
        tf.keras.backend.clear_session()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Set environment variables
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Enable both H100s
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
        
        # Get GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return False, {"error": "No GPUs available"}
            
        gpu_config = {
            "num_gpus": len(gpus),
            "memory_config": {}
        }
        
        # Clear and configure each GPU
        for gpu_index, gpu in enumerate(gpus):
            try:
                # Reset memory stats
                tf.config.experimental.reset_memory_stats(gpu)
                
                # Enable memory growth
                tf.config.experimental.set_memory_growth(gpu, True)
                
                # Get memory info after cleanup
                memory_info = tf.config.experimental.get_memory_info(f'GPU:{gpu_index}')
                available_memory = memory_info['current'] / 1e9  # Convert to GB
                
                # Configure memory limit based on available memory
                memory_limit = int(available_memory * 0.95)  # Use 95% of available memory
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=1024*memory_limit)]
                )
                
                # Store configuration details
                gpu_config["memory_config"][f"gpu_{gpu_index}"] = {
                    "available_memory_gb": available_memory,
                    "memory_limit_gb": memory_limit,
                    "growth_enabled": True
                }
                
            except RuntimeError as e:
                print(f"Warning: Error configuring GPU {gpu_index}: {e}")
                continue
                
        # Enable mixed precision for H100s
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            gpu_config["mixed_precision_enabled"] = True
        except Exception as e:
            print(f"Warning: Could not enable mixed precision: {e}")
            gpu_config["mixed_precision_enabled"] = False
            
        # Optional: PyTorch cleanup if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gpu_config["pytorch_cleanup"] = True
        except ImportError:
            gpu_config["pytorch_cleanup"] = False
            
        # Calculate optimal batch size based on available memory
        min_available_memory = min(config["available_memory_gb"] 
                                for config in gpu_config["memory_config"].values())
        
        if min_available_memory > 64.0:  # H100-level memory
            gpu_config["recommended_batch_size"] = 256_000
        elif min_available_memory > 32.0:
            gpu_config["recommended_batch_size"] = 128_000
        else:
            gpu_config["recommended_batch_size"] = 64_000
            
        return True, gpu_config
        
    except Exception as e:
        error_msg = f"Critical error during GPU configuration: {str(e)}"
        print(error_msg)
        return False, {"error": error_msg}
    
def monitor_gpu_memory(logger, gpu_config):
    """Monitor GPU memory usage"""
    if gpu_config and gpu_config.get('num_gpus', 0) > 0:
        for gpu_id in range(gpu_config['num_gpus']):
            try:
                memory_info = tf.config.experimental.get_memory_info(f'GPU:{gpu_id}')
                logger.info(f"GPU:{gpu_id} memory usage: {memory_info['current'] / 1e9:.2f} GB")
            except Exception as e:
                logger.warning(f"Could not monitor GPU:{gpu_id} memory: {e}")

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
        default=21_000_000, 
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

def adjust_batch_size(current_batch_size, gpu_config, logger):
    """Adjust batch size based on available GPU memory"""
    if gpu_config and 'memory_config' in gpu_config:
        try:
            # Get current memory usage
            memory_usage = max(
                tf.config.experimental.get_memory_info(f'GPU:{i}')['current'] / 1e9
                for i in range(gpu_config['num_gpus'])
            )
            
            # If using more than 90% of available memory, reduce batch size
            if memory_usage > 0.9 * gpu_config['memory_config']['gpu_0']['memory_limit_gb']:
                new_batch_size = max(1000, current_batch_size // 2)
                logger.warning(f"High memory usage detected. Reducing batch size from {current_batch_size} to {new_batch_size}")
                return new_batch_size
        except Exception as e:
            logger.warning(f"Error during batch size adjustment: {e}")
    
    return current_batch_size

def main():
    """
    Main entry point for MIMO dataset generation with enhanced GPU and memory management
    """
    try:
        # Configure GPU environment first and get configuration details
        success, gpu_config = configure_gpu_environment()
        
        # Parse arguments
        args = parse_arguments()
        
        # Initialize logger before any other operations
        logger = configure_logging(
            log_level=args.log_level,
            log_file=f'logs/mimo_dataset_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]'
        )

        if not success:
            logger.warning(f"GPU configuration failed: {gpu_config.get('error', 'Unknown error')}")
            logger.warning("Falling back to CPU")
        else:
            logger.info("GPU Configuration successful:")
            logger.info(f"Number of GPUs: {gpu_config['num_gpus']}")
            logger.info(f"Mixed precision enabled: {gpu_config['mixed_precision_enabled']}")
            for gpu_id, config in gpu_config['memory_config'].items():
                logger.info(f"\n{gpu_id}:")
                logger.info(f"  Available memory: {config['available_memory_gb']:.2f} GB")
                logger.info(f"  Memory limit: {config['memory_limit_gb']:.2f} GB")
        
        # Configure system parameters
        system_params = configure_system_parameters(args)
        system_params.replay_buffer_size = min(system_params.replay_buffer_size, 100000)
        
        # Set batch size based on GPU configuration if not specified
        if args.batch_size is None:
            if success:
                args.batch_size = gpu_config["recommended_batch_size"]
                logger.info(f"Using GPU-optimized batch size: {args.batch_size}")
            else:
                args.batch_size = 1000  # Conservative batch size for CPU
                logger.info(f"Using conservative batch size: {args.batch_size}")
        
        # Generate output path
        output_path = generate_output_path(args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Log configuration details
        logger.info("\n=== Configuration Details ===")
        logger.info(f"Total samples: {system_params.total_samples}")
        logger.info(f"TX Antennas: {system_params.num_tx}")
        logger.info(f"RX Antennas: {system_params.num_rx}")
        logger.info(f"Output path: {output_path}")
        logger.info(f"Verification enabled: {args.verify}")
        logger.info(f"Batch size: {args.batch_size}")

        # Monitor GPU memory before dataset generation
        if success:
            for gpu_id in range(gpu_config['num_gpus']):
                try:
                    memory_info = tf.config.experimental.get_memory_info(f'GPU:{gpu_id}')
                    logger.info(f"GPU:{gpu_id} initial memory usage: {memory_info['current'] / 1e9:.2f} GB")
                except Exception as e:
                    logger.warning(f"Could not monitor GPU:{gpu_id} memory: {e}")

        # Create and configure dataset generator
        logger.debug("Initializing dataset generator...")
        generator = MIMODatasetGenerator(
            system_params=system_params,
            logger=logger
        )
        
        # Generate dataset with enhanced error handling
        logger.info("Starting dataset generation...")
        try:
            # Monitor initial memory state
            monitor_gpu_memory(logger, gpu_config)

            generator.generate_dataset(
                num_samples=system_params.total_samples,
                save_path=output_path
            )
            monitor_gpu_memory(logger, gpu_config)
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
                # Reconfigure GPU environment before verification
                success, _ = configure_gpu_environment()
                if not success:
                    logger.warning("GPU reconfiguration failed before verification")
                
                if not os.path.exists(output_path):
                    logger.error("Dataset file does not exist")
                    return 1
                    
                file_size = os.path.getsize(output_path)
                if file_size == 0:
                    logger.error("Dataset file is empty")
                    return 1
                    
                logger.info(f"Dataset file size: {file_size / (1024*1024):.2f} MB")
                
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
        # Create a basic logger if the main logger hasn't been initialized
        if 'logger' not in locals():
            logger = logging.getLogger()
            logger.setLevel(logging.ERROR)
            handler = logging.StreamHandler()
            logger.addHandler(handler)
        
        logger.error(f"Critical error during execution: {str(e)}", exc_info=True)
        return 1
        
if __name__ == "__main__":
    try:
        # Set environment variables before any TF operations
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Set random seeds for reproducibility
        import numpy as np
        import random
        seed = 42
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        
        # Run main
        sys.exit(main())
    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)
