# Team Maintenance Agents - Punna Chowdhurry, Jane Slagle, and Diana Krmzian
# Tufts CS 138 - Final Project
# DeepQLearning.py

import numpy as np

class DeepQLearning:
    def __init__(self, env, gamma=0.99, alpha=0.01, epsilon=0.1):
        self.env = env
        self.num_acts = len(env.actions)
        self.state_size = env.state_size
        self.weights = np.random.randn(self.state_size, self.num_acts) * 0.01
        self.biases = np.zeros(self.num_acts)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon 
        self.all_acts_taken = []

    def choose_action(self, state):
        q_values = self._compute_q_values(state)

        if np.random.rand() < self.epsilon:
            action = self.env.actions[np.random.choice(len(self.env.actions))]
        else:
            action = self.env.actions[np.argmax(q_values)]

        self.all_acts_taken.append(action)

        return action

    def _compute_q_values(self, state):
        return np.dot(state, self.weights) + self.biases

    def update(self, state, action, reward, next_state, done):
        action_idx = self.env.actions.index(action)
        td_target = reward + self.gamma * (1 - done) * np.max(self._compute_q_values(next_state))
        td_error = td_target - self._compute_q_values(state)[action_idx]
        self.weights[:, action_idx] += self.alpha * td_error * state
        self.biases[action_idx] += self.alpha * td_error

    def train(self, num_episodes):
        rewards = []
        budgets = []
        deteriorations = []

        for _ in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            total_deterioration = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update(state, action, reward, next_state, done)

                deterioration = min(0, np.mean(next_state) - np.mean(state))
                total_deterioration += deterioration 
                total_reward += reward
                state = next_state

            rewards.append(total_reward)
            budgets.append(self.env.budget)
            deteriorations.append(total_deterioration) 

        return rewards, budgets, deteriorations
