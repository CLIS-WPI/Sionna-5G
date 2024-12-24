# Purpose of Training and Expected Results

This project focuses on developing a Reinforcement Learning (RL)-based model using the **Soft Actor-Critic (SAC)** algorithm to optimize beamforming decisions in a **Multiple-Input Multiple-Output (MIMO)** communication system. The model is trained on a dataset generated using **Sionna**, a simulation library for wireless communication systems, and a custom **Dataset Generator**. 

The dataset includes key information such as:
- **Channel States**: Representing the environment.
- **Actions**: Beamforming configurations.
- **Rewards**: Performance metrics such as spectral efficiency, throughput, and SINR.

The goal of this training is to maximize the efficiency and reliability of wireless communication in a **static MIMO environment** by dynamically selecting the best beamforming parameters. 

## Key Performance Indicators (KPIs)

The success of the training process and model performance is evaluated using the following KPIs:

1. **Spectral Efficiency (bits/s/Hz)**:
   - Measures the efficient utilization of the available bandwidth.
   - A higher spectral efficiency indicates optimal resource usage.

2. **Throughput (Mbps)**:
   - Represents the actual data transfer rate achieved by the system.
   - The model is trained to maximize throughput under varying channel conditions.

3. **Signal-to-Interference-Plus-Noise Ratio (SINR) (dB)**:
   - Reflects the quality of the communication signal compared to interference and noise.
   - Higher SINR values ensure better communication quality.

4. **Average Reward**:
   - Tracks the mean reward obtained by the RL agent per training episode.
   - Serves as a direct measure of the model’s ability to optimize the reward structure.

5. **Policy Entropy**:
   - Monitors the exploration behavior of the RL agent during training.
   - High entropy ensures diverse strategies are explored, contributing to robust performance.

6. **Validation Average Reward**:
   - Evaluates the model’s generalization ability using a validation dataset.
   - Ensures the model is not overfitting and performs well on unseen data.

## Expected Results

At the conclusion of the training process, the RL agent is expected to:
1. **Learn Optimal Beamforming Policies**: Dynamically select beamforming parameters that maximize spectral efficiency and throughput for different channel conditions.
2. **Enhance Communication Performance**: Demonstrate improved performance compared to traditional beamforming methods like fixed or random strategies.
3. **Converge on Key Metrics**: Show stability in metrics such as average reward, spectral efficiency, and SINR.
4. **Generalize Across Scenarios**: Maintain strong performance on unseen validation data, showcasing robustness and adaptability.

## Features of the Training Pipeline

1. **Dataset**:
   - The dataset used for training is generated using the **Sionna** library and includes 2.8 GB of data representing static MIMO environments.
   - Invalid samples (e.g., with FSPL < 20 dB) are filtered out during preprocessing to maintain data quality.

2. **Validation**:
   - The dataset is split into **training** (80%) and **validation** (20%) sets to evaluate the model’s ability to generalize beyond the training data.

3. **Logging and Monitoring**:
   - Training metrics such as actor loss, critic loss, average reward, and policy entropy are logged using TensorBoard.
   - Validation performance is checked periodically (every 5000 episodes) to ensure generalization.

4. **Scalability**:
   - Designed to handle large-scale datasets and leverage **GPU acceleration (2* h100)** for efficient training.
   - Mixed precision is enabled for better performance on supported GPUs.

5. **Model Saving**:
   - The model checkpoints are saved periodically during training to allow for resumption and further fine-tuning.

## Why This Project Matters

This project provides a robust framework for integrating **Reinforcement Learning** into wireless communication systems. By optimizing beamforming decisions, the trained model offers a practical solution to improve network performance, reliability, and resource utilization in real-world wireless communication scenarios.
