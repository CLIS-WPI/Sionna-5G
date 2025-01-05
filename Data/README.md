# MIMO Dataset Generator

## Overview
The **MIMO Dataset Generator** is a specialized tool designed to create datasets for training and evaluating AI models in **MIMO (Multiple-Input Multiple-Output)** communication systems. Leveraging the **Sionna library by NVIDIA**, this generator ensures realistic channel modeling, accurate path loss computations, and reliable inputs for reinforcement learning and optimization tasks in wireless communication research.

## Goals
- Generate datasets tailored for **explainable AI-driven beamforming optimization** in MIMO systems.
- Simplify dataset creation by focusing on essential components, ensuring practicality and ease of use.
- Produce high-quality datasets that balance computational efficiency with accurate training inputs.
- Facilitate integration into reinforcement learning pipelines with standardized dataset formats.

## Features
### Configurable MIMO Setup
- Flexible configurations (e.g., 4x4 MIMO).
- Support for **Uniform Linear Array (ULA)** and realistic antenna parameters (e.g., gain, spacing, orientation).

### Channel and Path Loss Modeling
- Incorporates realistic channel models (e.g., **Rayleigh block fading**).
- The path loss model used in this study is Free Space Path Loss (FSPL), chosen for its simplicity and suitability for static, line-of-sight scenarios. This choice allows for a focused evaluation of beamforming optimization. Future work may explore more complex models like Urban Micro (UMi) or Urban Macro (UMa) for extended scenarios.
- Includes **SNR customization** (0 to 30 dB) for varied environmental conditions.

### Robust Dataset Generation
- Outputs channel realizations with critical parameters like delay spread, coherence time, and Doppler shift.
- Generates large-scale datasets with metadata for easy analysis and validation.

### Validation
- **Tensor Shapes and Dimensions**:
  - Validate that tensor shapes and dimensions align with the specified MIMO configurations (e.g., antenna numbers, subcarriers, and channel paths).
  - Include assertions in the code to automatically check and log mismatches during dataset generation.

- **Critical Parameter Validation**:
  - Ensure all critical parameters, such as **FSPL**, **SNR**, and **path loss**, are within valid and realistic ranges.
  - Apply boundary checks during computation to prevent negative or invalid values (e.g., `assert FSPL >= 0` or `0 <= SNR <= 30 dB`).

- **Integrity Checks**:
  - Verify the dataset for inconsistencies, such as missing or corrupted entries.
  - Log warnings or errors if any anomalies are detected, and discard invalid samples to maintain dataset reliability.

- **Reproducibility**:
  - Set and document a fixed random seed (e.g., `np.random.seed(seed_value)`) to ensure consistent and reproducible dataset generation across runs.

- **Final Verification**:
  - Conduct a small-scale test run with a subset of configurations to confirm that the dataset meets expectations before generating the full dataset.


### Efficient Output Format
- Saves datasets in `.h5` format for efficient storage and integration into AI workflows.
- Metadata includes MIMO configurations, channel parameters, and dataset size.

## Scope
The **MIMO Dataset Generator** is focused solely on **dataset generation** for reinforcement learning and related tasks. Advanced metrics calculation, secondary utilities, and extended validation steps are deliberately excluded to maintain simplicity and practicality.

## Dataset Generation Workflow

#### Input Parameters
- **MIMO Configuration**:
  - Define the number of transmit (`num_tx`) and receive antennas (`num_rx`) (e.g., 4x4 MIMO).
  - Specify the number of streams (`num_streams`) and element spacing (`element_spacing`) for antenna arrays.
  - Configure carrier frequency (`carrier_frequency`), ensuring it falls within realistic ranges (e.g., 3.5 GHz).

- **Resource Grid Setup**:
  - Number of subcarriers (`num_subcarriers`) and OFDM symbols (`num_ofdm_symbols`) per frame.
  - Subcarrier spacing (`subcarrier_spacing`) based on 5G NR numerology.

- **Channel Parameters**:
  - Number of paths (`num_paths`) for multipath modeling.
  - Signal-to-Noise Ratio (`snr_range`) for varying environmental conditions.
  - Doppler shift and coherence time for realistic channel dynamics.
  - Path loss models (`path_loss_scenarios`) such as Free Space Path Loss (FSPL) or urban scenarios (UMi, UMa).

- **Modulation and Noise**:
  - Choose modulation schemes (`modulation_schemes`) from options like QPSK, 16QAM, or 64QAM.
  - Configure noise floor (`noise_floor`) for accurate signal modeling.

#### Validation
- **Tensor Shapes and Dimensions**:
  - Validate tensor shapes to ensure compatibility with the configured MIMO system (e.g., antenna arrays, streams, and channel responses).
  - Automatically log and resolve shape mismatches during generation.

- **Critical Parameter Integrity**:
  - Ensure no negative or invalid values in critical parameters such as FSPL, SNR, and path loss.
  - Include boundary checks to validate input ranges (`assert` statements for constraints like SNR ≥ 0).

- **Reproducibility**:
  - Set a fixed random seed (`random_seed`) for consistent and reproducible dataset generation.
  - Log seed details in the metadata for reference.

#### Output
- **Dataset Files**:
  - Save generated datasets in `.h5` format, optimized for storage and quick access.
  - Include metadata in each file detailing MIMO configurations, channel parameters, and generation settings.

- **Metadata**:
  - Comprehensive metadata to document input parameters, validation results, and random seed for reproducibility.

#### Example Configuration in `system_parameters.py`
Below is an example configuration snippet:
```python
from system_parameters import SystemParameters

# Example configuration
config = SystemParameters(
    num_tx=4,
    num_rx=4,
    carrier_frequency=3.5e9,
    num_subcarriers=64,
    num_ofdm_symbols=14,
    snr_range=(0.0, 30.0),
    modulation_schemes=['QPSK', '16QAM', '64QAM'],
    path_loss_scenarios=['fspl'],
    random_seed=42
)

print(config.get_config_dict())

###  Use Case
- Train reinforcement learning models (e.g., **Soft Actor-Critic**) for optimizing beamforming in adaptive MIMO systems.
- Use generated datasets to evaluate performance under static user scenarios, focusing on **spectral efficiency**, **SINR**, and **throughput**.

### Requirements
- **Python**: 3.8–3.11
- **Operating System**: Ubuntu 22.04 (recommended)
- **Libraries**:
  - TensorFlow (tested with TensorFlow 2.12)
  - Numpy
  - h5py
  - **Sionna library by NVIDIA**

