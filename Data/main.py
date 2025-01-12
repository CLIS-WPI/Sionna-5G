# main.py
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
import numpy as np
import random
from core.metrics_calculator import MetricsCalculator
import matplotlib.pyplot as plt
def configure_gpu_environment():
    """Configure GPU environment with memory growth and mixed precision."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return None, "No GPUs available. Falling back to CPU."

        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

        return {
            "num_gpus": len(gpus),
            "mixed_precision": True
        }, None
    except Exception as e:
        return None, f"GPU configuration error: {e}"

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="MIMO Dataset Generation Framework")

    parser.add_argument('--samples', type=int, default=1_320_000,
                        help="Number of samples to generate (default: 1,320,000)")
    parser.add_argument('--output', type=str, default=None,
                        help="Output path for dataset (default: dataset/mimo_dataset_TIMESTAMP.h5)")
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Logging level (default: INFO)")
    parser.add_argument('--verify', action='store_true', help="Verify generated dataset")
    parser.add_argument('--tx-antennas', type=int, default=4,
                        help="Number of transmit antennas (default: 4)")
    parser.add_argument('--rx-antennas', type=int, default=4,
                        help="Number of receive antennas (default: 4)")
    parser.add_argument('--batch-size', type=int, default=1000,
                        help="Batch size for dataset generation (default: 1000)")
    parser.add_argument('--carrier-freq', type=float, default=3.5e9,
                        help="Carrier frequency in Hz (default: 3.5GHz)")
    parser.add_argument('--num-streams', type=int, default=4,
                        help="Number of data streams (default: 4)")

    return parser.parse_args()

def generate_output_path(base_path=None):
    """Generate a unique output path for the dataset."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_path or f"dataset/mimo_dataset_{timestamp}.h5"

def main():
    """Main function for dataset generation."""
    args = parse_arguments()

    # Configure logging
    logger = configure_logging(
        log_level=args.log_level,
        log_file=f'logs/mimo_dataset_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]'
    )

    logger.info("Starting MIMO Dataset Generation...")

    # GPU configuration
    gpu_config, gpu_error = configure_gpu_environment()
    if gpu_config:
        logger.info(f"GPU Configuration: {gpu_config}")
    else:
        logger.warning(f"GPU Configuration Error: {gpu_error}. Using CPU.")

    # Configure system parameters
    system_params = SystemParameters(
        total_samples=args.samples,
        batch_size=args.batch_size,
        num_tx_antennas=args.tx_antennas,
        num_rx_antennas=args.rx_antennas,
        num_streams=args.num_streams,
        carrier_frequency=args.carrier_freq
    )

    # Initialize metrics calculator
    metrics_calc = MetricsCalculator(system_params)

    # Generate dataset
    output_path = generate_output_path(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        logger.info("Initializing MIMO Dataset Generator...")
        generator = MIMODatasetGenerator(
            system_params=system_params,
            logger=logger
        )

        logger.info(f"Generating dataset with {system_params.total_samples:,} samples...")
        dataset_path = generator.generate_dataset(save_path=output_path)
        
        if not dataset_path:
            logger.error("Dataset generation failed")
            sys.exit(1)

        logger.info(f"Dataset saved to {dataset_path}")
        generator.verify_complex_data(dataset_path)
        
        # Add metrics calculation here
        # In main.py, modify the metrics calculation section:

        # Replace the existing metrics calculation section with this:
        logger.info("Calculating and validating performance metrics...")
        with h5py.File(dataset_path, 'r') as f:
            try:
                # First, check if the data exists and log the available keys
                logger.info(f"Available dataset keys: {list(f.keys())}")
                
                # Extract necessary data from the dataset with proper group structure
                channel_response = tf.convert_to_tensor(f['channel_data']['channel_response'][:])
                tx_symbols = tf.convert_to_tensor(f['channel_data']['tx_symbols'][:])
                rx_symbols = tf.convert_to_tensor(f['channel_data']['rx_symbols'][:])
                snr_db = tf.convert_to_tensor(f['channel_data']['snr_db'][:])

                # Log shapes for debugging
                logger.info(f"Channel Response shape: {channel_response.shape}")
                logger.info(f"TX Symbols shape: {tx_symbols.shape}")
                logger.info(f"RX Symbols shape: {rx_symbols.shape}")
                logger.info(f"SNR shape: {snr_db.shape}")

                # Calculate metrics
                metrics = metrics_calc.calculate_enhanced_metrics(
                    channel_response=channel_response,
                    tx_symbols=tx_symbols,
                    rx_symbols=rx_symbols,
                    snr_db=snr_db
                )

                # Validate metrics separately
                validation_results = {}
                
                # Validate spectral efficiency
                if 'spectral_efficiency' in metrics:
                    se = metrics['spectral_efficiency']
                    se_valid = tf.reduce_mean(se) >= system_params.spectral_efficiency_min and \
                            tf.reduce_mean(se) <= system_params.spectral_efficiency_max
                    validation_results['spectral_efficiency'] = se_valid
                    
                # Validate effective SNR
                if 'effective_snr' in metrics:
                    eff_snr = metrics['effective_snr']
                    snr_valid = tf.reduce_mean(eff_snr) >= system_params.sinr_target
                    validation_results['effective_snr'] = snr_valid
                    
                # Validate condition number
                if 'condition_number' in metrics:
                    cond_num = metrics['condition_number']
                    cond_valid = tf.reduce_mean(cond_num) < 100  # Example threshold
                    validation_results['condition_number'] = cond_valid

                # Log validation results
                if all(validation_results.values()):
                    logger.info("[PASS] All performance targets met!")
                    for metric, value in metrics.items():
                        mean_value = tf.reduce_mean(value)
                        logger.info(f"{metric}: {mean_value:.4f}")
                else:
                    logger.warning("[FAIL] Some performance targets not met:")
                    for metric, is_valid in validation_results.items():
                        status = "[PASS]" if is_valid else "[FAIL]"
                        if metric in metrics:
                            mean_value = tf.reduce_mean(metrics[metric])
                            logger.warning(f"{status} {metric}: {mean_value:.4f}")
                        else:
                            logger.warning(f"{status} {metric}: N/A")
                # Calculate BER metrics
                ber_metrics = metrics_calc.calculate_ber(tx_symbols, rx_symbols, snr_db)

                # Log BER results
                logger.info("BER Performance Summary:")
                logger.info(f"Average BER: {ber_metrics['average_ber']:.2e}")
                logger.info(f"BER Target Met: {ber_metrics['all_targets_met']}")

                # Plot BER curve
                try:
                    plt.figure()
                    snr_points = sorted(ber_metrics['ber_curve'].keys())
                    ber_values = [ber_metrics['ber_curve'][snr] for snr in snr_points]
                    
                    plt.semilogy(snr_points, ber_values, 'b.-', label='Measured BER')
                    plt.grid(True)
                    plt.xlabel('SNR (dB)')
                    plt.ylabel('BER')
                    plt.title('BER vs SNR Performance')
                    plt.legend()
                    
                    # Create plots directory if it doesn't exist
                    os.makedirs('plots', exist_ok=True)
                    plt.savefig('plots/ber_performance.png')
                    logger.info("BER performance plot saved to plots/ber_performance.png")
                    plt.close()
                except Exception as e:
                    logger.warning(f"Failed to generate BER plot: {e}")

            except KeyError as e:
                logger.error(f"Failed to read dataset: Missing key {e}")
                raise
    except Exception as e:
        logger.error(f"Error during metrics calculation: {e}")
        raise

    logger.info("MIMO Dataset Generation completed successfully.")
    sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during execution: {e}")
        logging.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)