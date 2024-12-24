import os
import h5py
import tensorflow as tf
import numpy as np
from collections import deque
import random

# Replay Buffer Class
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def size(self):
        return len(self.buffer)

# Function to Get the Most Recent Dataset File
def get_latest_dataset(folder_path, extension=".h5"):
    files = [f for f in os.listdir(folder_path) if f.endswith(extension)]
    if not files:
        raise FileNotFoundError(f"No files with extension {extension} found in {folder_path}")
    # Get the most recent file based on creation time
    latest_file = max([os.path.join(folder_path, f) for f in files], key=os.path.getctime)
    return latest_file

# SAC Actor and Critic Networks
class Actor(tf.keras.Model):
    def __init__(self, action_dim):
        super(Actor, self).__init__()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.fc3 = tf.keras.layers.Dense(action_dim, activation='tanh')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.fc3(x)

class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.fc3 = tf.keras.layers.Dense(1)

    def call(self, state_action):
        x = self.fc1(state_action)
        x = self.fc2(x)
        return self.fc3(x)

# SAC Agent
class SACAgent:
    def __init__(self, state_dim, action_dim, replay_buffer, gamma=0.99, tau=0.005, alpha=0.2, actor_lr=3e-4, critic_lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Networks
        self.actor = Actor(action_dim)
        self.critic1 = Critic()
        self.critic2 = Critic()
        self.target_critic1 = Critic()
        self.target_critic2 = Critic()

        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic1_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.critic2_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        # Replay buffer
        self.replay_buffer = replay_buffer

        # Initialize target networks
        self.update_target_networks(1.0)

    def update_target_networks(self, tau):
        for target_param, param in zip(self.target_critic1.trainable_variables, self.critic1.trainable_variables):
            target_param.assign(tau * param + (1 - tau) * target_param)
        for target_param, param in zip(self.target_critic2.trainable_variables, self.critic2.trainable_variables):
            target_param.assign(tau * param + (1 - tau) * target_param)

    def select_action(self, state):
        state = np.expand_dims(state, axis=0)
        action = self.actor(state).numpy()[0]
        return action

    def train(self, batch_size):
        if self.replay_buffer.size() < batch_size:
            return

        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Train Critic Networks
        next_actions = self.actor(next_states)
        next_state_actions = tf.concat([next_states, next_actions], axis=1)
        target_q1 = self.target_critic1(next_state_actions)
        target_q2 = self.target_critic2(next_state_actions)
        target_q = tf.minimum(target_q1, target_q2)
        target_value = rewards + self.gamma * (1 - dones) * target_q

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            state_actions = tf.concat([states, actions], axis=1)
            q1 = self.critic1(state_actions)
            q2 = self.critic2(state_actions)
            critic1_loss = tf.reduce_mean(tf.square(target_value - q1))
            critic2_loss = tf.reduce_mean(tf.square(target_value - q2))

        critic1_grad = tape1.gradient(critic1_loss, self.critic1.trainable_variables)
        critic2_grad = tape2.gradient(critic2_loss, self.critic2.trainable_variables)

        self.critic1_optimizer.apply_gradients(zip(critic1_grad, self.critic1.trainable_variables))
        self.critic2_optimizer.apply_gradients(zip(critic2_grad, self.critic2.trainable_variables))

        # Train Actor Network
        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            new_state_actions = tf.concat([states, new_actions], axis=1)
            actor_loss = -tf.reduce_mean(self.critic1(new_state_actions) - self.alpha * tf.math.log(new_actions))

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        # Update Target Networks
        self.update_target_networks(self.tau)

# Main Training Loop
if __name__ == "__main__":
    # Find the latest dataset in the dataset folder
    dataset_folder = "dataset"
    latest_dataset = get_latest_dataset(dataset_folder)
    print(f"Loading dataset: {latest_dataset}")

    # Replay buffer
    replay_buffer = ReplayBuffer(max_size=20_000_000)

    # Load dataset from the latest .h5 file
    with h5py.File(latest_dataset, "r") as f:
        states = f["states"][:]
        actions = f["actions"][:]
        rewards = f["rewards"][:]
        next_states = f["next_states"][:]
        dones = f["dones"][:]

        # Add data to replay buffer
        for i in range(len(states)):
            replay_buffer.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

    # SAC Agent
    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    agent = SACAgent(state_dim, action_dim, replay_buffer)

    # Training loop
    batch_size = 256_000
    os.makedirs("models", exist_ok=True)  # Ensure model saving directory exists
    for episode in range(1_000_000):
        agent.train(batch_size)

        if episode % 1000 == 0:
            print(f"Episode {episode}: Training in progress...")

        # Save models every 10,000 episodes
        if episode % 10_000 == 0:
            agent.actor.save(f"models/actor_{episode}.h5")
            agent.critic1.save(f"models/critic1_{episode}.h5")
            agent.critic2.save(f"models/critic2_{episode}.h5")
            print(f"Saved models at episode {episode}")

    # Save final models after training is complete
    agent.actor.save("models/actor_final.h5")
    agent.critic1.save("models/critic1_final.h5")
    agent.critic2.save("models/critic2_final.h5")
    print("Training completed. Final models saved.")
