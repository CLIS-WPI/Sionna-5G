import os
import h5py
import tensorflow as tf
import numpy as np
from collections import deque
import random
import datetime
from typing import Tuple, Optional, Dict

# Configure GPU and memory growth
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
        
        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.trainable_variables))
        self.critic1_optimizer.apply_gradients(
            zip(critic1_grads, self.critic1.trainable_variables))
        self.critic2_optimizer.apply_gradients(
            zip(critic2_grads, self.critic2.trainable_variables))
        
        # Log metrics
        with self.train_summary_writer.as_default():
            tf.summary.scalar('actor_loss', actor_loss, step=self.actor_optimizer.iterations)
            tf.summary.scalar('critic1_loss', critic1_loss, step=self.critic1_optimizer.iterations)
            tf.summary.scalar('critic2_loss', critic2_loss, step=self.critic2_optimizer.iterations)
        
        return actor_loss, critic1_loss, critic2_loss

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

def main():
    # Configure GPU
    gpus = configure_gpu()
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    try:
        # Find the latest dataset
        dataset_folder = "dataset"
        latest_dataset = max(
            [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) 
            if f.endswith('.h5')],
            key=os.path.getctime
        )
        print(f"Loading dataset: {latest_dataset}")
        
        # Initialize replay buffer
        replay_buffer = ReplayBuffer(max_size=20_000_000)
        
        # Load dataset
        with h5py.File(latest_dataset, "r") as f:
            states = f["states"][:]
            actions = f["actions"][:]
            rewards = f["rewards"][:]
            next_states = f["next_states"][:]
            dones = f["dones"][:]
            
            # Add data to replay buffer
            for i in range(len(states)):
                replay_buffer.add(states[i], actions[i], rewards[i], 
                                next_states[i], dones[i])
        
        # Initialize SAC agent
        state_dim = states.shape[1]
        action_dim = actions.shape[1]
        agent = SACAgent(state_dim, action_dim, replay_buffer)
        
        # Get optimal batch size based on available GPU memory
        if gpus:
            try:
                memory_info = tf.config.experimental.get_memory_info('GPU:0')
                available_memory = memory_info['current'] / 1e9  # Convert to GB
                batch_size = get_optimal_batch_size(available_memory)
            except:
                batch_size = 64_000
        else:
            batch_size = 32_000
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Training loop
        for episode in range(1_000_000):
            try:
                losses = agent.train(batch_size)
                
                if episode % 1000 == 0:
                    print(f"Episode {episode}: Training in progress...")
                    if losses:
                        actor_loss, critic1_loss, critic2_loss = losses
                        print(f"Actor Loss: {actor_loss:.4f}, "
                            f"Critic1 Loss: {critic1_loss:.4f}, "
                            f"Critic2 Loss: {critic2_loss:.4f}")
                
                # Save models periodically
                if episode % 10_000 == 0:
                    agent.actor.save(f"models/actor_{episode}.h5")
                    agent.critic1.save(f"models/critic1_{episode}.h5")
                    agent.critic2.save(f"models/critic2_{episode}.h5")
                    print(f"Saved models at episode {episode}")
                    
            except tf.errors.ResourceExhaustedError:
                # Reduce batch size if OOM error occurs
                batch_size = max(1000, batch_size // 2)
                print(f"Reducing batch size to {batch_size} due to memory constraints")
                continue
                
            except Exception as e:
                print(f"Error during training: {str(e)}")
                continue
        
        # Save final models
        agent.actor.save("models/actor_final.h5")
        agent.critic1.save("models/critic1_final.h5")
        agent.critic2.save("models/critic2_final.h5")
        print("Training completed. Final models saved.")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()