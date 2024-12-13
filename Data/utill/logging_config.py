# utils/logging_config.py
# Advanced Logging Configuration Utility
# Manages comprehensive logging with multiple output options and flexible configuration
# Supports context-based logging, error tracking, and specialized logger management

import logging
import sys
from typing import Optional, Union, TextIO
import os
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