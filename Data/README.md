# MIMO Dataset Generator

## Overview
The **MIMO Dataset Generator** this part is designed to create datasets tailored for training and evaluating AI models in **MIMO (Multiple-Input Multiple-Output)** communication systems. Built with the **Sionna library from NVIDIA**, this tool ensures realistic channel modeling and accurate path loss calculations, providing reliable input for reinforcement learning and optimization in wireless communication research.

## Goals
- Provide datasets that support explainable AI-driven optimization in beamforming for MIMO systems.
- Simplify dataset generation by focusing on essential components without overcomplicating the implementation.
- Maintain high-quality datasets for efficient and accurate training while ensuring ease of integration into the broader training pipeline.

## Features
- **Configurable MIMO Setup**: Supports flexible configurations (e.g., 4x4 MIMO) with realistic antenna and channel models.
- **Path Loss Modeling**: Implements path loss calculations aligned with basic urban scenarios (e.g., Free Space Path Loss (FSPL), UMi, UMa) using the **Sionna library**.
- **Realistic Channel Responses**: Generates datasets with channel realizations based on Rayleigh block fading and defined SNR ranges.
- **Error and Integrity Validation**: Ensures dataset reliability through dimension checks, tensor validations, and no negative values for key parameters.
- **Reproducibility**: Supports fixed random seeds for consistent dataset generation.

## Scope
This tool is focused solely on **dataset generation** for training reinforcement learning models. Advanced metrics calculation, secondary utilities, and extended verification steps are excluded to avoid over-engineering. The generated dataset will serve as a foundation for training models with optimal quality and simplicity.

## Dataset Generation Steps
1. **Input Parameters**:
   - Define MIMO setup: Antenna configurations, modulation order, and SNR range.
   - Specify path loss model: Basic urban scenarios or FSPL.
   - Configure channel parameters: Number of paths, coherence time, and Doppler shift.

2. **Validation**:
   - Ensure tensor shapes and dimensions match expected MIMO configurations.
   - Prevent negative or invalid values in parameters like FSPL and SNR.

3. **Output**:
   - Generate dataset files in `.h5` format.
   - Include metadata for MIMO configuration, channel parameters, and dataset size.

## Requirements
- Python 3.8–3.11
- Recommended OS: Ubuntu 22.04
- Libraries:
  - TensorFlow (tested with TensorFlow 2.12)
  - Numpy
  - h5py
  - **Sionna library from NVIDIA**

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mimo-dataset-generator.git
   cd mimo-dataset-generator
