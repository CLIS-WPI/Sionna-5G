import os
import h5py
import tensorflow as tf
import numpy as np
from collections import deque
import random
import datetime
from typing import Tuple, Optional, Dict
from tqdm import trange
import time
import logging
import os
import tensorflow as tf
# Configure GPU and memory growth
from gpu_config import configure_gpu
def configure_gpu():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Using GPU for processing.")
            # Enable mixed precision
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
        return gpus
    except Exception as e:
        print(f"Error configuring GPU: {str(e)}")
        return None
    
def filter_invalid_samples(dataset):
    """
    Enhanced filtering with comprehensive validation.
    """
    fspl = dataset['fspl']
    
    # Multiple validation conditions
    valid_indices = np.where(
        (fspl >= 20.0) &  # Minimum valid path loss
        (fspl <= 160.0) &  # Maximum valid path loss
        (fspl != 0.0) &    # Exclude zero values
        np.isfinite(fspl)  # Exclude inf/nan values
    )[0]

    # Print detailed statistics
    total_samples = len(fspl)
    zero_samples = np.sum(fspl == 0.0)
    below_min = np.sum((fspl > 0.0) & (fspl < 20.0))
    above_max = np.sum(fspl > 160.0)
    invalid_values = np.sum(~np.isfinite(fspl))

    print(f"\nDataset Filtering Statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Zero values: {zero_samples} ({zero_samples/total_samples*100:.2f}%)")
    print(f"Below 20 dB: {below_min} ({below_min/total_samples*100:.2f}%)")
    print(f"Above 160 dB: {above_max} ({above_max/total_samples*100:.2f}%)")
    print(f"Invalid values: {invalid_values} ({invalid_values/total_samples*100:.2f}%)")
    print(f"Valid samples: {len(valid_indices)} ({len(valid_indices)/total_samples*100:.2f}%)")

    # Filter dataset
    filtered_dataset = {
        key: dataset[key][valid_indices] for key in dataset.keys()
    }

    return filtered_dataset


# Replay Buffer Class with enhanced memory management
class ReplayBuffer:
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)
        
    def add(self, state: np.ndarray, action: np.ndarray, 
            reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.float32),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def size(self) -> int:
        return len(self.buffer)

# Actor Network with GPU optimization
class Actor(tf.keras.Model):
    def __init__(self, action_dim: int):
        super(Actor, self).__init__()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.fc3 = tf.keras.layers.Dense(action_dim, activation='tanh')
        
    @tf.function
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.fc3(x)

# Critic Network with GPU optimization
class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.fc3 = tf.keras.layers.Dense(1)
        
    @tf.function
    def call(self, state_action):
        x = self.fc1(state_action)
        x = self.fc2(x)
        return self.fc3(x)

# SAC Agent with enhanced training capabilities
class SACAgent:
    def __init__(self, state_dim: int, action_dim: int, replay_buffer: ReplayBuffer,
                gamma: float = 0.99, tau: float = 0.005, alpha: float = 0.2,
                actor_lr: float = 3e-4, critic_lr: float = 3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # Initialize networks
        self.actor = Actor(action_dim)
        self.critic1 = Critic()
        self.critic2 = Critic()
        self.target_critic1 = Critic()
        self.target_critic2 = Critic()
        
        # Initialize optimizers with gradient clipping
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr, clipnorm=1.0)
        self.critic1_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr, clipnorm=1.0)
        self.critic2_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr, clipnorm=1.0)
        
        self.replay_buffer = replay_buffer
        self.update_target_networks(1.0)
        
        # Initialize tensorboard writer
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = f'logs/gradient_tape/{current_time}'
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)

    @tf.function
    def update_target_networks(self, tau):
        for target_param, param in zip(self.target_critic1.trainable_variables,
                                    self.critic1.trainable_variables):
            target_param.assign(tau * param + (1 - tau) * target_param)
        for target_param, param in zip(self.target_critic2.trainable_variables,
                                    self.critic2.trainable_variables):
            target_param.assign(tau * param + (1 - tau) * target_param)

    @tf.function
    def select_action(self, state):
        state = tf.expand_dims(state, axis=0)
        action = self.actor(state)
        return action[0]

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape(persistent=True) as tape:
            # Actor loss
            new_actions = self.actor(states)
            new_state_actions = tf.concat([states, new_actions], axis=1)
            actor_loss = -tf.reduce_mean(
                self.critic1(new_state_actions) - self.alpha * tf.math.log(new_actions)
            )
            
            # Critic loss
            next_actions = self.actor(next_states)
            next_state_actions = tf.concat([next_states, next_actions], axis=1)
            target_q1 = self.target_critic1(next_state_actions)
            target_q2 = self.target_critic2(next_state_actions)
            target_q = tf.minimum(target_q1, target_q2)
            target_value = rewards + self.gamma * (1 - dones) * target_q
            
            state_actions = tf.concat([states, actions], axis=1)
            q1 = self.critic1(state_actions)
            q2 = self.critic2(state_actions)
            critic1_loss = tf.reduce_mean(tf.square(target_value - q1))
            critic2_loss = tf.reduce_mean(tf.square(target_value - q2))

        # Apply gradients
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        critic1_grads = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        critic2_grads = tape.gradient(critic2_loss, self.critic2.trainable_variables)
        
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic1_optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_variables))
        self.critic2_optimizer.apply_gradients(zip(critic2_grads, self.critic2.trainable_variables))
        
        # Compute additional metrics
        avg_reward = tf.reduce_mean(rewards)  # Average reward
        policy_entropy = -tf.reduce_mean(tf.math.log(new_actions + 1e-8))  # Policy entropy

        # Log metrics
        with self.train_summary_writer.as_default():
            tf.summary.scalar('actor_loss', actor_loss, step=self.actor_optimizer.iterations)
            tf.summary.scalar('critic1_loss', critic1_loss, step=self.critic1_optimizer.iterations)
            tf.summary.scalar('critic2_loss', critic2_loss, step=self.critic2_optimizer.iterations)
            tf.summary.scalar('avg_reward', avg_reward, step=self.actor_optimizer.iterations)
            tf.summary.scalar('policy_entropy', policy_entropy, step=self.actor_optimizer.iterations)
        
        return actor_loss, critic1_loss, critic2_loss, avg_reward, policy_entropy

    def validate(self, validation_data):
        """Evaluate the model on validation data."""
        states, actions, rewards, next_states, dones = validation_data
        with tf.GradientTape() as tape:
            next_actions = self.actor(next_states)
            next_state_actions = tf.concat([next_states, next_actions], axis=1)
            target_q1 = self.target_critic1(next_state_actions)
            target_q2 = self.target_critic2(next_state_actions)
            target_q = tf.minimum(target_q1, target_q2)
            predicted_rewards = rewards + self.gamma * (1 - dones) * target_q

        avg_validation_reward = tf.reduce_mean(predicted_rewards)
        return avg_validation_reward.numpy()
    
    def train(self, batch_size: int):
        if self.replay_buffer.size() < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        actor_loss, critic1_loss, critic2_loss = self.train_step(
            tf.convert_to_tensor(states, dtype=tf.float32),
            tf.convert_to_tensor(actions, dtype=tf.float32),
            tf.convert_to_tensor(rewards, dtype=tf.float32),
            tf.convert_to_tensor(next_states, dtype=tf.float32),
            tf.convert_to_tensor(dones, dtype=tf.float32)
        )
        
        self.update_target_networks(self.tau)
        return actor_loss, critic1_loss, critic2_loss

def get_optimal_batch_size(available_memory: float) -> int:
    if available_memory > 32.0:
        return 256_000
    elif available_memory > 16.0:
        return 128_000
    else:
        return 64_000

# Main function (Modified to include validation set)
def main():
    # Configure logging
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename="logs/log.txt",
        filemode="a",
        format="%(asctime)s - %(message)s",
        level=logging.INFO
    )

    # Log to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    gpus = configure_gpu()
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    try:
        dataset_folder = os.path.expanduser("~/Desktop/milad-5G/dataset/Sionna-5G/Data/dataset")
        try:
            latest_dataset = max(
                [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith('.h5')],
                key=os.path.getctime
            )
            logging.info(f"Loading dataset: {latest_dataset}")
        except ValueError:
            raise FileNotFoundError(f"No .h5 files found in the dataset folder: {dataset_folder}")

        with h5py.File(latest_dataset, "r") as f:
            raw_dataset = {
                'states': f["states"][:],
                'actions': f["actions"][:],
                'rewards': f["rewards"][:],
                'next_states': f["next_states"][:],
                'dones': f["dones"][:],
                'fspl': f["fspl"][:]
            }

        filtered_dataset = filter_invalid_samples(raw_dataset)

        if len(filtered_dataset['states']) < 1000:
            raise ValueError("Too many samples filtered out. Check dataset quality.")

        # Split dataset into training and validation sets
        split_idx = int(0.8 * len(filtered_dataset['states']))
        training_data = {
            key: value[:split_idx] for key, value in filtered_dataset.items()
        }
        validation_data = {
            key: value[split_idx:] for key, value in filtered_dataset.items()
        }

        # Initialize replay buffer
        replay_buffer = ReplayBuffer(max_size=15_000_000)
        for i in range(len(training_data['states'])):
            replay_buffer.add(
                training_data['states'][i], 
                training_data['actions'][i], 
                training_data['rewards'][i], 
                training_data['next_states'][i], 
                training_data['dones'][i]
            )

        state_dim = training_data['states'].shape[1]
        action_dim = training_data['actions'].shape[1]
        agent = SACAgent(state_dim, action_dim, replay_buffer)

        if gpus:
            try:
                memory_info = tf.config.experimental.get_memory_info('GPU:0')
                available_memory = memory_info['current'] / 1e9
                batch_size = get_optimal_batch_size(available_memory)
            except:
                batch_size = 64_000
        else:
            batch_size = 32_000

        os.makedirs("checkpoints/models", exist_ok=True)

        # Import time and start tracking
        from tqdm import trange
        import time

        start_time = time.time()

        for episode in trange(1_000_000, desc="Training Progress", unit="episode"):
            try:
                losses = agent.train(batch_size)

                if episode % 1000 == 0:
                    elapsed_time = time.time() - start_time
                    avg_time_per_episode = elapsed_time / (episode + 1)
                    remaining_time = avg_time_per_episode * (1_000_000 - episode - 1)

                    logging.info(f"Episode {episode}: Avg Time/Episode: {avg_time_per_episode:.2f}s, "
                                f"Estimated Remaining Time: {remaining_time / 3600:.2f}h")

                    if losses:
                        actor_loss, critic1_loss, critic2_loss, avg_reward, policy_entropy = losses
                        logging.info(f"Actor Loss: {actor_loss:.4f}, Critic1 Loss: {critic1_loss:.4f}, "
                                    f"Critic2 Loss: {critic2_loss:.4f}, Avg Reward: {avg_reward:.4f}, "
                                    f"Policy Entropy: {policy_entropy:.4f}")

                if episode % 5000 == 0:
                    val_reward = agent.validate((
                        tf.convert_to_tensor(validation_data['states'], dtype=tf.float32),
                        tf.convert_to_tensor(validation_data['actions'], dtype=tf.float32),
                        tf.convert_to_tensor(validation_data['rewards'], dtype=tf.float32),
                        tf.convert_to_tensor(validation_data['next_states'], dtype=tf.float32),
                        tf.convert_to_tensor(validation_data['dones'], dtype=tf.float32)
                    ))
                    logging.info(f"Episode {episode}: Validation Avg Reward: {val_reward:.4f}")

                if episode % 10_000 == 0:
                    agent.actor.save(f"checkpoints/models/actor_{episode}.h5")
                    agent.critic1.save(f"checkpoints/models/critic1_{episode}.h5")
                    agent.critic2.save(f"checkpoints/models/critic2_{episode}.h5")
                    logging.info(f"Saved models at episode {episode}")

            except tf.errors.ResourceExhaustedError:
                batch_size = max(1000, batch_size // 2)
                logging.info(f"Reducing batch size to {batch_size} due to memory constraints")
                continue

            except Exception as e:
                logging.error(f"Error during training: {str(e)}")
                continue

        agent.actor.save("checkpoints/models/actor_final.h5")
        agent.critic1.save("checkpoints/models/critic1_final.h5")
        agent.critic2.save("checkpoints/models/critic2_final.h5")
        logging.info("Training completed. Final models saved.")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")

