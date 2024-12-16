import h5py
import numpy as np
import os

# Define expected ranges and metrics
EXPECTED_RANGES = {
    "ber": {"range": (0, 1)},  # Bit Error Rate
    "sinr": {"range": (0, 40)},  # SINR in dB
    "spectral_efficiency": {"range": (0, 10)},  # Spectral Efficiency
    "throughput": {"range": (0, 100)},  # Throughput percentage
    "effective_snr": {"range": (-10, 30)},  # Effective SNR
    "eigenvalues": {"shape": (-1, 4)},  # Channel Eigenvalues
    "channel_response": {"shape": (-1, 4, 4)},  # Channel Matrix
    "path_loss_data/fspl": {"range": (20, 160)},  # Path Loss
    "path_loss_data/scenario_pathloss": {"range": (20, 160)},  # Scenario-Specific Path Loss
}

# Function to display statistics for numerical features
def check_statistics(data, feature_name, expected_range=None, expected_shape=None):
    print(f"\n--- Validating {feature_name} ---")
    print(f"Shape: {data.shape}")
    print(f"Min: {np.min(data)}, Max: {np.max(data)}")
    print(f"Mean: {np.mean(data)}, Variance: {np.var(data)}")

    # Check range
    if expected_range:
        min_val, max_val = expected_range
        invalid = np.where((data < min_val) | (data > max_val))
        if len(invalid[0]) > 0:
            print(f"Warning: {len(invalid[0])} values out of range [{min_val}, {max_val}].")
        else:
            print(f"All values within the range [{min_val}, {max_val}].")
    
    # Check shape
    if expected_shape:
        if data.shape[1:] != expected_shape[1:]:
            print(f"Warning: Shape mismatch. Expected: {expected_shape}, Found: {data.shape}")

# Validate features inside modulation_data
def validate_modulation_data(group):
    for mod_scheme in group:
        print(f"\n--- Modulation Scheme: {mod_scheme} ---")
        for feature in group[mod_scheme]:
            if feature in EXPECTED_RANGES:
                data = np.array(group[mod_scheme][feature])
                check_statistics(data, f"{mod_scheme}/{feature}", 
                                 expected_range=EXPECTED_RANGES[feature].get("range"),
                                 expected_shape=EXPECTED_RANGES[feature].get("shape"))

# Validate path loss data
def validate_path_loss(group):
    for path_feature in group:
        full_key = f"path_loss_data/{path_feature}"
        if full_key in EXPECTED_RANGES:
            data = np.array(group[path_feature])
            check_statistics(data, full_key, expected_range=EXPECTED_RANGES[full_key]["range"])

# Main validation function
def validate_dataset(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    print(f"Reading Dataset: {file_path}")
    with h5py.File(file_path, 'r') as f:
        print("\n--- Dataset Structure ---")
        f.visit(print)  # Print the structure of the HDF5 file
        
        # Validate modulation data
        if "modulation_data" in f:
            validate_modulation_data(f["modulation_data"])
        else:
            print("Error: 'modulation_data' group is missing!")

        # Validate path loss data
        if "path_loss_data" in f:
            validate_path_loss(f["path_loss_data"])
        else:
            print("Error: 'path_loss_data' group is missing!")

# File Path to Dataset
file_path = "dataset/mimo_dataset_20241216_172003.h5"

# Run Validation
if __name__ == "__main__":
    validate_dataset(file_path)
