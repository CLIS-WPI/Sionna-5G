# utils/tensor_shape_validator.py
# TensorFlow Tensor Shape Validation Utility
# Provides robust shape checking, broadcasting, and normalization for complex tensors
# Ensures tensor compatibility and provides detailed error reporting for machine learning workflows

import tensorflow as tf
from typing import Union, List, Any

def assert_tensor_shape(
    tensor: tf.Tensor, 
    expected_shape: List[Union[int, None]], 
    tensor_name: str = 'input_tensor'
) -> tf.Tensor:
    """
    Robust tensor shape validation with detailed error reporting
    
    Args:
        tensor (tf.Tensor): Input tensor to validate
        expected_shape (List[Union[int, None]]): Expected shape dimensions
        tensor_name (str, optional): Name of tensor for error reporting
    
    Returns:
        tf.Tensor: Validated input tensor
    
    Raises:
        ValueError: If tensor shape doesn't match expected shape
    """
    try:
        actual_shape = tensor.shape.as_list()
        
        # Validate rank (number of dimensions)
        if len(actual_shape) != len(expected_shape):
            raise ValueError(
                f"Rank mismatch for {tensor_name}: "
                f"Expected {len(expected_shape)} dimensions, "
                f"Got {len(actual_shape)} dimensions"
            )
        
        # Check each dimension
        for idx, (expected_dim, actual_dim) in enumerate(zip(expected_shape, actual_shape)):
            if expected_dim is not None and expected_dim != actual_dim:
                raise ValueError(
                    f"Shape mismatch for {tensor_name} at dimension {idx}: "
                    f"Expected {expected_dim}, Got {actual_dim}"
                )
        
        return tensor
    
    except Exception as e:
        tf.print(f"❌ Shape validation error for {tensor_name}:")
        tf.print(f"   Actual shape: {tensor.shape}")
        tf.print(f"   Expected shape: {expected_shape}")
        tf.print(f"   Error: {str(e)}")
        raise

def broadcast_tensors(
    tensor1: tf.Tensor, 
    tensor2: tf.Tensor
) -> tuple:
    """
    Broadcast two tensors to compatible shapes
    
    Args:
        tensor1 (tf.Tensor): First input tensor
        tensor2 (tf.Tensor): Second input tensor
    
    Returns:
        tuple: Broadcasted tensors
    """
    try:
        # Determine the maximum rank between the two tensors
        max_rank = max(len(tensor1.shape), len(tensor2.shape))
        
        # Pad shapes with 1s to match rank
        shape1 = [1] * (max_rank - len(tensor1.shape)) + tensor1.shape.as_list()
        shape2 = [1] * (max_rank - len(tensor2.shape)) + tensor2.shape.as_list()
        
        # Check broadcasting compatibility
        for dim1, dim2 in zip(shape1, shape2):
            if dim1 != 1 and dim2 != 1 and dim1 != dim2:
                raise ValueError("Tensors cannot be broadcasted")
        
        # Use TensorFlow's broadcast_to
        return (
            tf.broadcast_to(tensor1, tf.math.maximum(tensor1.shape, tensor2.shape)),
            tf.broadcast_to(tensor2, tf.math.maximum(tensor1.shape, tensor2.shape))
        )
    
    except Exception as e:
        tf.print("❌ Tensor broadcasting error:")
        tf.print(f"   Tensor 1 shape: {tensor1.shape}")
        tf.print(f"   Tensor 2 shape: {tensor2.shape}")
        tf.print(f"   Error: {str(e)}")
        raise

def normalize_complex_tensor(
    tensor: tf.Tensor, 
    axis: Union[int, List[int]] = None
) -> tf.Tensor:
    """
    Normalize complex tensor with flexible axis handling
    
    Args:
        tensor (tf.Tensor): Complex input tensor
        axis (Union[int, List[int]], optional): Axis/axes for normalization
    
    Returns:
        tf.Tensor: Normalized complex tensor
    """
    # Default to all axes if not specified
    if axis is None:
        axis = list(range(len(tensor.shape)))
    
    # Compute magnitude
    magnitude = tf.abs(tensor)
    
    # Compute normalization factor
    norm_factor = tf.reduce_mean(magnitude, axis=axis, keepdims=True)
    
    # Convert norm_factor to complex before adding epsilon
    norm_factor = tf.cast(norm_factor, dtype=tf.complex64)
    epsilon = tf.complex(1e-10, 0.0)
    
    # Avoid division by zero with complex epsilon
    return tensor / (norm_factor + epsilon)

def validate_tensor_properties(
    tensor: tf.Tensor, 
    dtype: tf.DType = None, 
    min_rank: int = None, 
    max_rank: int = None
) -> tf.Tensor:
    """
    Comprehensive tensor property validation
    
    Args:
        tensor (tf.Tensor): Input tensor to validate
        dtype (tf.DType, optional): Expected data type
        min_rank (int, optional): Minimum tensor rank
        max_rank (int, optional): Maximum tensor rank
    
    Returns:
        tf.Tensor: Validated input tensor
    """
    # Check data type
    if dtype is not None and tensor.dtype != dtype:
        raise TypeError(
            f"Incorrect tensor dtype. "
            f"Expected {dtype}, Got {tensor.dtype}"
        )
    
    # Check tensor rank
    rank = len(tensor.shape)
    if min_rank is not None and rank < min_rank:
        raise ValueError(
            f"Tensor rank too low. "
            f"Expected at least {min_rank}, Got {rank}"
        )
    
    if max_rank is not None and rank > max_rank:
        raise ValueError(
            f"Tensor rank too high. "
            f"Expected at most {max_rank}, Got {rank}"
        )
    
    return tensor

# Example usage demonstrating utility functions
def example_usage():
    # Shape validation
    x = tf.random.normal([32, 4, 4])
    assert_tensor_shape(x, [32, 4, 4], 'example_tensor')
    
    # Tensor broadcasting
    a = tf.random.normal([32, 1, 4])
    b = tf.random.normal([32, 4, 1])
    broadcasted_a, broadcasted_b = broadcast_tensors(a, b)
    
    # Complex tensor normalization
    complex_tensor = tf.complex(
        tf.random.normal([32, 4, 4]), 
        tf.random.normal([32, 4, 4])
    )
    normalized_tensor = normalize_complex_tensor(complex_tensor)