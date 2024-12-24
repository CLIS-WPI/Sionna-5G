import os
import tensorflow as tf

# Set environment variables to control GPU memory growth and logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # Use first GPU

def configure_gpu():
    try:
        # Enable memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"Memory growth enabled for GPU: {gpu}")
                except RuntimeError as e:
                    print(f"Error setting memory growth: {e}")
                    continue
            
            # Set memory limit (optional, adjust based on your GPU)
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=1024*8)]  # 8GB limit
                )
            except RuntimeError as e:
                print(f"Error setting memory limit: {e}")
            
            print(f"Found {len(gpus)} GPU(s). Using GPU for processing.")
            
            # Enable mixed precision
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("Mixed precision policy set to mixed_float16")
            except Exception as e:
                print(f"Error setting mixed precision: {e}")
        
        return gpus
    except Exception as e:
        print(f"Error configuring GPU: {str(e)}")
        return None

if __name__ == "__main__":
    # Test GPU configuration
    gpus = configure_gpu()
    if gpus:
        print("GPU configuration successful")
        # Print available GPU memory
        try:
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            print(f"Available GPU memory: {memory_info['available'] / 1e9:.2f} GB")
        except Exception as e:
            print(f"Could not get memory info: {e}")
    else:
        print("No GPUs available, will run on CPU")