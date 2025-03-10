# utils/logging_config.py
# Advanced Logging Configuration Utility
# Manages comprehensive logging with multiple output options and flexible configuration
# Supports context-based logging, error tracking, and specialized logger management

import logging
import sys
from typing import Optional, Union, TextIO
import os
from datetime import datetime
import tensorflow as tf
import numpy as np 

def configure_logging(
    log_level: Union[int, str] = logging.INFO,
    log_file: Optional[str] = None,
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    console_output: bool = True,
    output_stream: Optional[TextIO] = sys.stdout
) -> logging.Logger:
    """
    Configures comprehensive logging with multiple output options

    Args:
        log_level (Union[int, str]): Logging level (e.g., logging.INFO, 'DEBUG')
        log_file (Optional[str]): Path to log file for file logging
        log_format (str): Custom log message format
        console_output (bool): Enable console logging
        output_stream (TextIO): Output stream for console logging

    Returns:
        logging.Logger: Configured root logger
    """
    # Convert string log levels to numeric values if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear any existing handlers
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(log_format)

    # Console Handler
    if console_output:
        console_handler = logging.StreamHandler(output_stream)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File Handler (if log file specified)
    if log_file:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except IOError as e:
            logger.error(f"Could not create log file: {e}")

    return logger

class LoggerManager:
    """
    Advanced logger management with context support
    """
    
    @staticmethod
    def get_logger(
        name: Optional[str] = None, 
        log_level: Union[int, str] = logging.INFO
    ) -> logging.Logger:
        """
        Get a configured logger with optional name
        
        Args:
            name (Optional[str]): Logger name
            log_level (Union[int, str]): Logging level
        
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        return logger
    
    @staticmethod
    def log_exception(
        logger: logging.Logger, 
        message: str, 
        exc_info: bool = True
    ):
        """
        Log an exception with optional additional message
        
        Args:
            logger (logging.Logger): Logger instance
            message (str): Additional context message
            exc_info (bool): Include exception traceback
        """
        logger.exception(message, exc_info=exc_info)

def create_dataset_logger(
    dataset_name: str, 
    log_file: Optional[str] = 'dataset_generation.log'
) -> logging.Logger:
    """
    Create a specialized logger for dataset generation
    
    Args:
        dataset_name (str): Name of the dataset
        log_file (Optional[str]): Path to log file
    
    Returns:
        logging.Logger: Configured dataset logger
    """
    logger = configure_logging(
        log_level=logging.INFO,
        log_file=log_file,
        log_format=f'%(asctime)s - {dataset_name} - %(levelname)s - %(message)s'
    )
    return logger

def create_channel_logger(log_dir: str = 'logs/channel') -> logging.Logger:
    """
    Creates a specialized logger for channel-related debugging
    
    Args:
        log_dir (str): Directory for channel logs
    
    Returns:
        logging.Logger: Configured channel logger
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'channel_debug_{datetime.now():%Y%m%d_%H%M%S}.log')
    
    logger = logging.getLogger('channel_debug')
    logger.setLevel(logging.DEBUG)
    
    # Detailed formatter for channel analysis
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s'
    )
    
    # File handler with detailed debug information
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler with less verbose output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def create_ber_logger(log_dir: str = 'logs/ber') -> logging.Logger:
    """
    Creates a specialized logger for BER analysis
    
    Args:
        log_dir (str): Directory for BER logs
    
    Returns:
        logging.Logger: Configured BER logger
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'ber_analysis_{datetime.now():%Y%m%d_%H%M%S}.log')
    
    logger = logging.getLogger('ber_analysis')
    logger.setLevel(logging.DEBUG)
    
    # Detailed formatter for BER metrics
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - SNR:%(snr).2f - BER:%(ber).2e - %(message)s'
    )
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class MIMOLogger:
    """
    Centralized logging management for MIMO system components
    """
    def __init__(self, base_log_dir: str = 'logs'):
        self.base_log_dir = base_log_dir
        self.channel_logger = create_channel_logger(os.path.join(base_log_dir, 'channel'))
        self.ber_logger = create_ber_logger(os.path.join(base_log_dir, 'ber'))
        self.logger = self.channel_logger  # Use channel_logger for general logging

    # Add this new method
    def log_generation_progress(self, batch_idx: int, total_batches: int, current_metrics: dict):
        """
        Log dataset generation progress and current batch metrics.
        
        Args:
            batch_idx (int): Current batch index
            total_batches (int): Total number of batches
            current_metrics (dict): Dictionary containing current batch metrics
        """
        progress_pct = (batch_idx + 1) / total_batches * 100
        
        # Log progress
        self.channel_logger.info(
            f"Generation Progress: {progress_pct:.1f}% ({batch_idx + 1}/{total_batches} batches)"
        )
        
        # Log summary of current metrics
        if current_metrics:
            metrics_summary = {
                key: float(tf.reduce_mean(value).numpy()) 
                for key, value in current_metrics.items() 
                if isinstance(value, (tf.Tensor, np.ndarray))
            }
            self.channel_logger.debug(f"Batch {batch_idx} metrics: {metrics_summary}")

    def log_channel_stats(self, channel_response, snr):
        """Log channel response statistics."""
        try:
            # Use tf.math.reduce_std instead of tf.std
            stats = {
                'magnitude_mean': tf.reduce_mean(tf.abs(channel_response)).numpy(),
                'magnitude_std': tf.math.reduce_std(tf.abs(channel_response)).numpy(),
                'phase_mean': tf.reduce_mean(tf.math.angle(channel_response)).numpy(),
                'phase_std': tf.math.reduce_std(tf.math.angle(channel_response)).numpy(),
                'snr_mean': tf.reduce_mean(snr).numpy(),
                'snr_std': tf.math.reduce_std(snr).numpy()
            }
            
            # Use channel_logger instead of self.logger
            self.channel_logger.info(f"Channel Statistics: {stats}")
            return stats
            
        except Exception as e:
            self.channel_logger.error(f"Error calculating channel statistics: {str(e)}")
            raise
        
    def log_ber_measurement(self, ber_value, snr_db, additional_info=None):
        """Log BER measurements with context"""
        context = {
            'ber': float(ber_value),
            'snr': float(snr_db)
        }
        
        if additional_info:
            context.update(additional_info)
            
        self.ber_logger.debug(
            "BER Measurement",
            extra=context
        )

# Example usage
def main():
    # Basic logging configuration
    logger = configure_logging(
        log_level='DEBUG', 
        log_file='app.log'
    )
    
    # Example logging
    logger.info("Application started")
    logger.debug("Debug information")
    
    try:
        # Simulated error
        raise ValueError("Example error")
    except Exception as e:
        LoggerManager.log_exception(logger, "An error occurred")

if __name__ == "__main__":
    main()