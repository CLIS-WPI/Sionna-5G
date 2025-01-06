# utils/tensor_shape_validator.py
# TensorFlow Tensor Shape Validation Utility
# Provides robust shape checking, broadcasting, and normalization for complex tensors
# Ensures tensor compatibility and provides detailed error reporting for machine learning workflows

import tensorflow as tf
import sionna
from sionna.mimo import StreamManagement
from typing import Dict, List, Union, Any, Optional, Tuple
import logging

def validate_mimo_tensor_shapes(
    channel_response: tf.Tensor,
    num_tx_antennas: int,
    num_rx_antennas: int,
    batch_size: int,
    name: str = "channel_response"
) -> bool:
    """
    Validate MIMO channel response tensor shapes
    
    Args:
        channel_response (tf.Tensor): MIMO channel response tensor
        num_tx_antennas (int): Number of transmit antennas
        num_rx_antennas (int): Number of receive antennas
        batch_size (int): Batch size
        name (str): Tensor name for error reporting
        
    Returns:
        bool: True if shapes are valid
    """
    try:
        expected_shape = [batch_size, num_rx_antennas, num_tx_antennas]
        actual_shape = channel_response.shape.as_list()
        
        if len(actual_shape) != 3:
            raise ValueError(
                f"{name} must have 3 dimensions [batch_size, num_rx, num_tx], "
                f"got shape {actual_shape}"
            )
            
        if actual_shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for {name}: "
                f"Expected {expected_shape}, got {actual_shape}"
            )
            
        return True
        
    except Exception as e:
        logging.error(f"MIMO tensor shape validation failed: {str(e)}")
        return False

def validate_mimo_metrics(
    metrics: Dict[str, tf.Tensor],
    batch_size: int,
    num_tx_antennas: int,
    num_rx_antennas: int
) -> bool:
    """
    Validate shapes of MIMO metrics tensors
    
    Args:
        metrics (Dict[str, tf.Tensor]): Dictionary of MIMO metrics
        batch_size (int): Batch size
        num_tx_antennas (int): Number of transmit antennas
        num_rx_antennas (int): Number of receive antennas
        
    Returns:
        bool: True if all metrics shapes are valid
    """
    try:
        expected_shapes = {
            'spectral_efficiency': [batch_size],
            'effective_snr': [batch_size],
            'condition_number': [batch_size],
            'eigenvalues': [batch_size, min(num_rx_antennas, num_tx_antennas)]
        }
        
        for metric_name, tensor in metrics.items():
            if metric_name not in expected_shapes:
                logging.warning(f"Unexpected metric: {metric_name}")
                continue
                
            expected_shape = expected_shapes[metric_name]
            actual_shape = tensor.shape.as_list()
            
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch for {metric_name}: "
                    f"Expected {expected_shape}, got {actual_shape}"
                )
                
        return True
        
    except Exception as e:
        logging.error(f"MIMO metrics validation failed: {str(e)}")
        return False

def validate_stream_management(
    stream_manager: StreamManagement,
    num_tx_antennas: int,
    num_rx_antennas: int
) -> bool:
    """
    Validate Sionna StreamManagement configuration
    
    Args:
        stream_manager: Sionna StreamManagement object
        num_tx_antennas (int): Number of transmit antennas
        num_rx_antennas (int): Number of receive antennas
        
    Returns:
        bool: True if configuration is valid
    """
    try:
        # Verify stream configuration
        num_streams = stream_manager.num_streams_per_rx
        max_streams = min(num_tx_antennas, num_rx_antennas)
        
        if num_streams > max_streams:
            raise ValueError(
                f"Number of streams ({num_streams}) cannot exceed "
                f"min(num_tx, num_rx) = {max_streams}"
            )
            
        return True
        
    except Exception as e:
        logging.error(f"Stream management validation failed: {str(e)}")
        return False

def validate_channel_frequencies(
    frequencies: tf.Tensor,
    num_subcarriers: int,
    subcarrier_spacing: float
) -> bool:
    """
    Validate channel frequency configuration
    
    Args:
        frequencies (tf.Tensor): Frequency points tensor
        num_subcarriers (int): Number of subcarriers
        subcarrier_spacing (float): Subcarrier spacing in Hz
        
    Returns:
        bool: True if frequency configuration is valid
    """
    try:
        expected_shape = [num_subcarriers]
        actual_shape = frequencies.shape.as_list()
        
        if actual_shape != expected_shape:
            raise ValueError(
                f"Frequency shape mismatch: Expected {expected_shape}, "
                f"got {actual_shape}"
            )
            
        # Verify frequency spacing
        freq_diff = tf.experimental.numpy.diff(frequencies)
        if not tf.reduce_all(tf.abs(freq_diff - subcarrier_spacing) < 1e-6):
            raise ValueError("Inconsistent subcarrier spacing")
            
        return True
        
    except Exception as e:
        logging.error(f"Frequency validation failed: {str(e)}")
        return False