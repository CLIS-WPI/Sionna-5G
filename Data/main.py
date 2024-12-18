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

os.makedirs('logs', exist_ok=True)

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
        default=12_000, 
        help="Number of samples to generate (default: 1,000,000)"
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
    default_path = f"dataset/mimo_dataset_{timestamp}.h5"
    return base_path or default_path

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
