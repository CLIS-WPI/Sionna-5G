import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def validate_dataset(dataset_file):
    """
    Validate the dataset features against expected ranges and generate statistics.

    Args:
        dataset_file (str): Path to the dataset file (e.g., CSV file).

    Returns:
        None
    """
    # Load the dataset
    print(f"Loading dataset from {dataset_file}...")

    # Expected ranges and categories based on the simulation plan
    expected_ranges = {
        "SINR (dB)": (-5, 25),
        "Spectral Efficiency (bits/s/Hz)": (4, 8),
        "Throughput (Mbps)": (0, 100),  # Example range
        "Modulation Order": [4, 16, 64],  # QPSK, 16-QAM, 64-QAM
        "SNR (dB)": (-10, 30),
    }

    print("\n--- Dataset Validation ---")
    for feature, expected_range in expected_ranges.items():
        if feature in dataset.columns:
            data = dataset[feature]
            print(f"\nFeature: {feature}")
            print(f"Min: {data.min()}, Max: {data.max()}")
            print(f"Mean: {data.mean()}, Std: {data.std()}")

            # Check range or categories
            if isinstance(expected_range, tuple):
                out_of_range = data[(data < expected_range[0]) | (data > expected_range[1])]
                if not out_of_range.empty:
                    print(f"Warning: {len(out_of_range)} values out of range {expected_range}.")
            elif isinstance(expected_range, list):
                invalid_values = data[~data.isin(expected_range)]
                if not invalid_values.empty:
                    print(f"Warning: {len(invalid_values)} invalid values. Allowed: {expected_range}.")

            # Plot histogram
            plt.hist(data, bins=30, alpha=0.7)
            plt.title(f"Distribution of {feature}")
            plt.xlabel(feature)
            plt.ylabel("Frequency")
            plt.grid()
            plt.show()
        else:
            print(f"Feature {feature} not found in dataset!")

    print("\nValidation Complete.")


def check_plan_alignment(plan_features, dataset_file):
    """
    Check alignment between simulation plan features and dataset features.

    Args:
        plan_features (list): Features expected from the simulation plan.
        dataset_file (str): Path to the dataset file (e.g., CSV file).

    Returns:
        None
    """
    # Load the dataset
    print(f"Loading dataset from {dataset_file}...")
    dataset = pd.read_csv(dataset_file)
    dataset_features = dataset.columns.tolist()

    # Check alignment
    print("\n--- Alignment Check ---")
    missing_features = [f for f in plan_features if f not in dataset_features]
    extra_features = [f for f in dataset_features if f not in plan_features]

    if missing_features:
        print(f"Missing Features: {missing_features}")
    else:
        print("No features missing from the simulation plan!")

    if extra_features:
        print(f"Extra Features: {extra_features}")
    else:
        print("No extra features found in the dataset!")


if __name__ == "__main__":
    # Path to the dataset file
    dataset_file = "mimo_dataset.csv"  # Replace with your dataset file path

    # Expected features based on the simulation plan
    plan_features = [
        "SINR (dB)",
        "Spectral Efficiency (bits/s/Hz)",
        "Throughput (Mbps)",
        "Modulation Order",
        "SNR (dB)",
        "Antenna Gain",
        "Element Spacing",
    ]

    # Step 1: Validate dataset values and ranges
    validate_dataset(dataset_file)

    # Step 2: Check alignment between simulation plan and dataset features
    check_plan_alignment(plan_features, dataset_file)
