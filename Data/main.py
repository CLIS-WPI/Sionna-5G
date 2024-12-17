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
        default=1_000_000, 
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
from integrity.dataset_integrity_checker import MIMODatasetIntegrityChecker  # Add this import
os.makedirs('logs', exist_ok=True)

def main():
    """
    Main entry point for MIMO dataset generation
    
    Returns:
        int: Exit status (0 for success, 1 for failure)
    """
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Configure logging
        logger = configure_logging(
            log_level=args.log_level, 
            log_file=f'logs/mimo_dataset_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        # Configure system parameters
        system_params = configure_system_parameters(args)
        
        # Generate output path
        output_path = generate_output_path(args.output)
        
        # Create dataset generator
        generator = MIMODatasetGenerator(
            system_params=system_params, 
            logger=logger
        )
        
        # Log start of dataset generation
        logger.info(f"Starting MIMO dataset generation")
        logger.info(f"Total samples: {system_params.total_samples}")
        logger.info(f"Output path: {output_path}")
        logger.info(f"TX Antennas: {system_params.num_tx}")
        logger.info(f"RX Antennas: {system_params.num_rx}")
        
        # Generate dataset
        generator.generate_dataset(
            num_samples=system_params.total_samples, 
            save_path=output_path
        )
        
        # Verify dataset if requested
        if args.verify:
            logger.info("Verifying generated dataset...")
            try:
                # Create integrity checker instance with the dataset path
                with MIMODatasetIntegrityChecker(output_path) as checker:
                    integrity_report = checker.check_dataset_integrity()
                    
                    if integrity_report['overall_status']:
                        logger.info("Dataset verification successful")
                        # Log detailed statistics if needed
                        for mod_scheme, mod_details in integrity_report['modulation_schemes'].items():
                            logger.info(f"\n{mod_scheme} Statistics:")
                            logger.info(f"  Samples: {mod_details['samples']}")
                            logger.info(f"  Integrity: {'✓ VALID' if mod_details['integrity'] else '✗ INVALID'}")
                    else:
                        logger.warning("Dataset verification failed")
                        logger.debug(f"Integrity report: {integrity_report}")
            except Exception as e:
                logger.error(f"Dataset verification error: {str(e)}")
                return 1
        
        logger.info("MIMO dataset generation completed successfully")
        return 0
    
    except Exception as e:
        logging.error(f"Critical error during dataset generation: {e}")
        logging.exception("Detailed error traceback:")
        return 1

if __name__ == "__main__":
    sys.exit(main())