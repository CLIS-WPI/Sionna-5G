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
### Input Parameters
- Configure MIMO setup: Define number of antennas (e.g., 4x4), modulation order, and SNR range.
- Specify channel parameters: Number of paths, coherence time, Doppler shift, and path loss models.
- Include polarization options and array orientation for diverse scenarios.

### Validation
- Check tensor shapes, dimensions, and integrity of generated datasets.
- Ensure no negative or invalid values in critical parameters (e.g., FSPL, SNR).

### Output
- Dataset files saved in `.h5` format.
- Metadata embedded for clarity and reproducibility.

##  Use Case
- Train reinforcement learning models (e.g., **Soft Actor-Critic**) for optimizing beamforming in adaptive MIMO systems.
- Use generated datasets to evaluate performance under static user scenarios, focusing on **spectral efficiency**, **SINR**, and **throughput**.

## Requirements
- **Python**: 3.8–3.11
- **Operating System**: Ubuntu 22.04 (recommended)
- **Libraries**:
  - TensorFlow (tested with TensorFlow 2.12)
  - Numpy
  - h5py
  - **Sionna library by NVIDIA**
