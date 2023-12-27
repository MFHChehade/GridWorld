import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class QLearning:

    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, infinite_horizon=False):
        """
        Initialize a QLearning object.

        Parameters:
        - env: The environment.
        - alpha (float): The learning rate (default: 0.1).
        - gamma (float): The discount factor (default: 0.9).
        - epsilon (float): The probability of choosing a random action (default: 0.1).
        - infinite_horizon (bool): Whether the problem is infinite horizon (default: False).
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((env.n_states, env.n_actions))
        self.infinite_horizon = infinite_horizon
    
    def reset(self):
        """
        Reset the Q-values.
        """
        self.Q = np.zeros((self.env.n_states, self.env.n_actions))

    # Method to choose an action based on epsilon-greedy policy
    def choose_action(self, state):
        """
        Choose an action based on the epsilon-greedy policy.

        Parameters:
        - state: The current state.

        Returns:
        - action: The chosen action.
        """
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.env.n_actions)
        else:
            return np.random.choice(np.flatnonzero(self.Q[state] == self.Q[state].max()))
    
    # Method to update Q-values based on experience
    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-values.

        Parameters:
        - state: The current state.
        - action: The chosen action.
        - reward: The reward received.
        - next_state: The next state.
        """
        if not self.infinite_horizon and done:
            self.Q[state, action] += self.alpha * (reward - self.Q[state, action])
        else:
            self.Q[state, action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action])
    
    # Method to train the Q-learning agent
    def train(self, n_episodes=1000, max_steps=1000, verbose=False):
        """
        Train the agent.

        Parameters:
        - n_episodes (int): The number of episodes (default: 1000).
        - max_steps (int): The maximum number of steps per episode (default: 1000).
        - verbose (bool): If True, displays a progress bar; if False, no progress bar is shown (default: False).
        """
        reward_per_episode = []
        steps_per_episode = []
        episodes = range(n_episodes)
        self.reset()

        if verbose:
            tqdm.write(f"Training for {n_episodes} episodes...")
            episodes = tqdm(episodes, desc='Episodes', unit=' episodes')

        for episode in episodes:
            state = self.env.reset()
            total_reward = 0
            steps = 0 
            for _ in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps +=1
                if done and self.infinite_horizon == False:
                    break
            # print(steps, total_reward)
            # self.env.render()
            reward_per_episode.append(total_reward)
            steps_per_episode.append(steps)

        return [reward / steps for reward, steps in zip(reward_per_episode, steps_per_episode)]

 
    # Method to plot the moving average of rewards per episode
    def plot_moving_average_rewards(self, rewards_per_episode, window_size=200):
        """
        Plot the moving average of rewards per episode.

        Parameters:
        - rewards_per_episode (list): The rewards per episode.
        - window_size (int): The size of the moving average window (default: 200).
        """

        # Calculate moving average
        moving_average = pd.Series(rewards_per_episode).rolling(window=window_size, min_periods=1).mean()

        # Plot the moving average
        plt.figure(figsize=(8, 6))
        plt.plot(moving_average)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Average Reward per step')
        plt.grid(True)
        plt.show()
    
    # Method to test the trained agent
    def test(self, n_episodes=1000, max_steps=1000):
        """
        Test the agent.

        Parameters:
        - n_episodes (int): The number of episodes (default: 1000).
        - max_steps (int): The maximum number of steps per episode (default: 1000).

        Returns:
        - rewards (list): The rewards per episode.
        """
        self.env.reset()
        rewards = []
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            for _ in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                if done and self.infinite_horizon == False:
                    break
            rewards.append(episode_reward)
        return rewards
    
    # Methods to retrieve policy, value function, and Q-values
    def get_policy(self):
        """
        Get the policy.

        Returns:
        - policy (list): The policy.
        """
        policy = []
        for state in range(self.env.n_states):
            policy.append(np.argmax(self.Q[state]))
        return np.array(policy)
    
    def get_value_function(self):
        """
        Get the value function.

        Returns:
        - value_function (list): The value function.
        """
        value_function = []
        for state in range(self.env.n_states):
            value_function.append(np.max(self.Q[state]))
        return np.array(value_function)
    
    def get_q_values(self):
        """
        Get the Q-values.

        Returns:
        - Q (list): The Q-values.
        """
        return self.Q
