# Team Maintenance Agents - Punna Chowdhurry, Jane Slagle, and Diana Krmzian
# Tufts CS 138 - Final Project
# DeepSARSA.py

import numpy as np

class DeepSARSA:
   """ 
   SARSA ("State-Action-Reward-State-Action") is an on-policy RL algorithm that updates the action-value 
   function based on the action taken by the current policy. It uses the current state, action, reward,
   next state, and next state to improve policy. 

   Deep SARSA integrates deep learning techniques to approximate the action-value function (Q-function). 
   Instead of using a simple table to store Q-values, Deep SARSA uses a neural network to handle complex and 
   continuous state spaces, making it suitable for tasks with large infinite state-action spaces.
   """
   def __init__(self, env, gamma=0.99, alpha=0.01, epsilon=0.1, epsilon_decay=0.9995, min_epsilon=0.01, hidden_size=128): #Increased hidden layer to work better with the complex env
       self.env = env
       self.gamma = gamma
       self.alpha = alpha
       self.epsilon = epsilon
       self.epsilon_decay = epsilon_decay
       self.min_epsilon = min_epsilon
       self.state_size = env.state_size
       self.action_size = len(env.actions)

       #Use a simple 2 layer neural network and initialize weights and biases to it
       #Input values for the first hidden layer 
       self.weights1 = np.random.randn(self.state_size, hidden_size) * 0.01
       self.bias1 = np.zeros(hidden_size)
       #Input values from the first hidden layer to the 2nd hidden layer 
       self.weights2 = np.random.randn(hidden_size, hidden_size) * 0.01
       self.bias2 = np.zeros(hidden_size)
       #Input values from the first hidden layer to the 2nd hidden layer to the output layer (Q-values)
       self.weights3 = np.random.randn(hidden_size, self.action_size) * 0.01
       self.bias3 = np.zeros(self.action_size)

       self.all_acts_taken = []  

   def forward(self, state):
       """
       Compute q-values by forward passing through the network. This will help predict the action-value function Q(s,a) for given state s 
       """
       z1 = np.dot(state, self.weights1) + self.bias1
       #This is the ReLU activation. Adds non-linear activation for neural networks, so the agent can learn non-linear actions/decision 
       #f(x) = max(0,x)
       a1 = np.maximum(0, z1)   
       z2 = np.dot(a1, self.weights2) + self.bias2
       #This is the ReLU activation
       #f(x) = max(0,x)
       a2 = np.maximum(0, z2)   
       q_values = np.dot(a2, self.weights3) + self.bias3
       return q_values, (state, z1, a1, z2, a2)

   def backward(self, cache, td_error, action):
       """
       Backward pass to update the neural network weights and biases based on td error
       """
       state, z1, a1, z2, a2 = cache
       #Gradients of q-values 
       q_grad = np.zeros_like(self.bias3)
       q_grad[action] = td_error

       grad_w3 = np.outer(a2, q_grad)
       grad_b3 = q_grad

       #Backpropagate through the 2nd hidden layer (z2, a2)
       #delta2 is the gradient of the loss bc of z2 
       delta2 = np.dot(q_grad, self.weights3.T)
       delta2[z2 <= 0] = 0   
       grad_w2 = np.outer(a1, delta2)
       grad_b2 = delta2

       #Backpropagate through the 1st hidden layer
       delta1 = np.dot(delta2, self.weights2.T)
       delta1[z1 <= 0] = 0   
       grad_w1 = np.outer(state, delta1)
       grad_b1 = delta1

       #Weights and biases updates with learning rate 
       self.weights3 += self.alpha * grad_w3
       self.bias3 += self.alpha * grad_b3
       self.weights2 += self.alpha * grad_w2
       self.bias2 += self.alpha * grad_b2
       self.weights1 += self.alpha * grad_w1
       self.bias1 += self.alpha * grad_b1

   def choose_action(self, state):
       """
       Epsilon-greedy policy will be applied to help the agent choose an action
       This will balances both exploitation and exploration 
       """
       if np.random.rand() < self.epsilon:
           #This is the exploration stage 
           action = np.random.choice(self.action_size)
       else:
           #This is the exploitation stage, using the neural network to predict q-values 
           q_values, _ = self.forward(state)
           #Get the action with the highest q-values 
           action = np.argmax(q_values)
       self.all_acts_taken.append(self.env.actions[action])   
       return action

   def train(self, num_episodes):
       """
       The agent will be trained using all the components of Deep SARSA 
       """
       rewards = []
       budgets = []
       deteriorations = []

       for ep in range(num_episodes):
           state = self.env.reset()
           total_reward = 0
           total_deterioration = 0
           done = False
           action = self.choose_action(state)

           while not done:
               next_state, reward, done = self.env.step(self.env.actions[action], action_duration=1)
               next_action = self.choose_action(next_state)

               deterioration = min(0, np.mean(next_state) - np.mean(state))
               total_deterioration += deterioration 

               #Adjust as needed to improve results. Increasing the min and max range will get more variations of rewards 
               #Compare results with (-10, 10) vs. (-20, 20)
               reward = max(-20, min(20, reward))   
               reward += 1  

               #Calculate TD error, which is the difference between the current Q-value and the updated Q-value based on the reward and future estimate
               #TD error = TD Target - Q(s, a) 
               q_values, cache = self.forward(state)
               next_q_values, _ = self.forward(next_state)
               td_target = reward + self.gamma * next_q_values[next_action] * (1 - done)
               td_error = td_target - q_values[action]

               self.backward(cache, td_error, action)

               state, action = next_state, next_action
               total_reward += reward

           rewards.append(total_reward)
           budgets.append(self.env.budget)
           deteriorations.append(total_deterioration) 

           self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

       return rewards, budgets, deteriorations
