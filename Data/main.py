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
from core.path_loss_model import PathLossManager

def configure_gpu_environment(system_params=None):
    """Enhanced GPU configuration with SystemParameters integration"""
    try:
        # Basic GPU setup
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return False, {"error": "No GPUs available"}
            
        # Configure memory fraction from system_params
        memory_fraction = system_params.max_memory_fraction if system_params else 0.8
        
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                # Calculate memory limit based on system_params
                memory_limit = int(tf.config.experimental.get_memory_info(f'GPU:0')['total'] 
                                 * memory_fraction)
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
                
        # Enhanced strategy configuration
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
        )
        
        # Configure mixed precision
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        return True, {
            "num_gpus": len(gpus),
            "strategy": strategy,
            "memory_fraction": memory_fraction,
            "mixed_precision": True
        }
    except Exception as e:
        return False, {"error": str(e)}
    
def validate_system_configuration(system_params, gpu_config, logger):
    """Enhanced validation with SystemParameters integration"""
    try:
        # Validate basic parameters
        if system_params.total_samples <= 0:
            raise ValueError("Total samples must be positive")
            
        # Validate batch size against GPU memory
        if gpu_config and gpu_config.get('num_gpus', 0) > 0:
            max_batch_size = system_params.max_batch_size
            min_batch_size = system_params.min_batch_size
            
            if system_params.batch_size > max_batch_size:
                logger.warning(f"Batch size {system_params.batch_size} exceeds maximum {max_batch_size}")
                system_params.batch_size = max_batch_size
                
            if system_params.batch_size < min_batch_size:
                logger.warning(f"Batch size {system_params.batch_size} below minimum {min_batch_size}")
                system_params.batch_size = min_batch_size
                
        return True
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False

def monitor_memory_usage(system_params, gpu_config, logger):
    """Enhanced memory monitoring"""
    try:
        for gpu_id in range(gpu_config.get('num_gpus', 0)):
            memory_info = tf.config.experimental.get_memory_info(f'GPU:{gpu_id}')
            current_usage = memory_info['current'] / 1e9
            total_memory = memory_info['total'] / 1e9
            usage_percent = (current_usage / total_memory) * 100
            
            logger.info(f"GPU:{gpu_id} - Usage: {current_usage:.2f}GB/{total_memory:.2f}GB ({usage_percent:.1f}%)")
            
            # Adjust batch size if needed
            if usage_percent > 90:
                system_params.batch_size = max(
                    system_params.min_batch_size,
                    int(system_params.batch_size * 0.8)
                )
                logger.warning(f"Reducing batch size to {system_params.batch_size}")
                
    except Exception as e:
        logger.warning(f"Memory monitoring failed: {e}")

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
        default=1_320_000, 
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

def configure_system_parameters(args, gpu_config):
    """Enhanced system parameters configuration"""
    try:
        # Initialize with command line arguments
        system_params = SystemParameters(
            num_tx=args.tx_antennas,
            num_rx=args.rx_antennas,
            total_samples=args.samples
        )
        
        # Adjust batch size based on GPU configuration
        if gpu_config and gpu_config.get('num_gpus', 0) > 0:
            recommended_batch_size = min(
                system_params.max_batch_size,
                gpu_config.get('recommended_batch_size', 32000)
            )
            system_params.batch_size = recommended_batch_size
            
        # Adjust replay buffer size based on available memory
        system_params.replay_buffer_size = min(
            system_params.replay_buffer_size,
            100000 * gpu_config.get('num_gpus', 1)
        )
        
        return system_params
    except Exception as e:
        raise RuntimeError(f"Failed to configure system parameters: {e}")

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
    try:
        # GPU setup first
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        strategy = tf.distribute.MirroredStrategy()
        
        # Parse args and initialize components
        args = parse_arguments()
        logger = configure_logging(
            log_level=args.log_level,
            log_file=f'logs/mimo_dataset_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]'
        )
        
        # Setup gpu_config
        success = True if gpus else False
        gpu_config = {
            "num_gpus": len(gpus),
            "strategy": strategy,
            "mixed_precision_enabled": True,
            "memory_config": {},
            "recommended_batch_size": 32000 if success else 1000
        }

        # Configure system and validate
        system_params = configure_system_parameters(args, gpu_config)
        system_params.replay_buffer_size = min(system_params.replay_buffer_size, 100000)
        
        if not validate_system_configuration(system_params, gpu_config, logger):
            logger.error("System configuration validation failed")
            return 1

        # Setup paths and batch size
        output_path = generate_output_path(args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if not success:
            logger.warning(f"GPU configuration failed: {gpu_config.get('error', 'Unknown error')}")
            logger.warning("Falling back to CPU")
        else:
            logger.info("GPU Configuration successful:")
            logger.info(f"Number of GPUs: {gpu_config['num_gpus']}")
            logger.info(f"Mixed precision enabled: {gpu_config['mixed_precision_enabled']}")

        # Set batch size based on GPU configuration if not specified
        if args.batch_size is None:
            args.batch_size = gpu_config["recommended_batch_size"] if success else 1000
            logger.info(f"Using {'GPU-optimized' if success else 'conservative'} batch size: {args.batch_size}")
        
        # Log configuration details
        logger.info("\n=== Configuration Details ===")
        logger.info(f"Total samples: {system_params.total_samples}")
        logger.info(f"TX Antennas: {system_params.num_tx}")
        logger.info(f"RX Antennas: {system_params.num_rx}")
        logger.info(f"Output path: {output_path}")
        logger.info(f"Verification enabled: {args.verify}")
        logger.info(f"Batch size: {args.batch_size}")

        # Generate dataset with enhanced error handling
        logger.info("Starting dataset generation...")
        try:
            # Monitor initial memory state
            monitor_memory_usage(system_params, gpu_config, logger)

            # Create path loss manager instance
            path_loss_manager = PathLossManager(system_params)

            # Generate distances for path loss calculation
            distances = tf.random.uniform(
                [system_params.batch_size],
                minval=1.0,  # Minimum distance (1 meter)
                maxval=500.0,  # Maximum distance (500 meters)
                dtype=tf.float32
            )

            # Calculate path loss
            path_loss = path_loss_manager.calculate_path_loss(
                distances,
                scenario='umi'  # or 'uma' based on your requirements
            )

            # Create and configure dataset generator with path loss
            generator = MIMODatasetGenerator(
                system_params=system_params,
                logger=logger,
                path_loss_manager=path_loss_manager
            )
            
            # Generate dataset with path loss
            generator.generate_dataset(
                num_samples=system_params.total_samples,
                save_path=output_path,
                path_loss=path_loss
            )

            monitor_memory_usage(system_params, gpu_config, logger)
            logger.info("Dataset generation completed")
            
        except Exception as e:
            logger.error(f"Dataset generation failed: {str(e)}", exc_info=True)
            return 1

        # Verification section (if enabled)
        if args.verify:
            logger.info("Starting dataset verification...")
            try:
                with MIMODatasetIntegrityChecker(output_path) as checker:
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
