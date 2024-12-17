# Team Maintenance Agents - Punna Chowdhurry, Jane Slagle, and Diana Krmzian
# Tufts CS 138 - Final Project
# SMDP.py

import numpy as np
import random

class SMDP:
    """
    Implements a Q-learning based SMDP agent that can be paried with the InfraPlanner environment.
    SMDP algorithms are distinct in that they allow for actions with variable time step lengths. We 
    implement this by maintaining and updating Q values scaled by the time duration for each action.
    """
    def __init__(self, env, gamma = 0.9, alpha = 0.1, epsilon = 0.1):
        self.env = env
        self.num_acts = len(env.actions)
        self.q_vals = np.zeros(self.num_acts)
        self.all_acts_taken = []
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def choose_act(self):
        """
        choose action for inputted state using eps-greedy policy approach
        """
        #explore w/ prob eps = means take rand act
        if np.random.uniform() < self.epsilon:
            #get the corresp action out of our list of actions from the env
            action = self.env.actions[self.choose_rand_act()]
        else:
            action = self.env.actions[self.choose_best_act()]
        
        #now that have taken this action, need to keep track that we took it for our results by adding it to our all_acts_taken list!!
        self.all_acts_taken.append(action)

        return action

    def choose_rand_act(self):
        """
        helper func for choose_act, randomly selects an act (explores) in eps-greedy act selection approach
        """
        return random.choice(range(self.num_acts))  
        
    def choose_best_act(self):
        """
        helper func for choose_act, selects act w/ highest est value (exploits)
        """
        return np.argmax(self.q_vals)

    def calc_q_val(self, action, reward, action_duration):
        """
        calculates the q value for the inputted action based on the normal updated Q(s,a) func where 
        Q(s,a) += alpha * [reward at current action + gamma * q value for action that gives max t that state - current act q value]
        and we also incorporate how action_duration should impact it
        """
        #calc based off equation specified in docstrings above
        act = self.env.actions.index(action)  #get out which specific act from all poss acts are taking here so that can udpate the q val for it

        #need scale the discount factor by action_duration since time spent on action will impact the q val to accurately depict how much time spent
        #on action should discount its q val
        self.q_vals[act] += self.alpha * (reward + (self.gamma ** action_duration) * (np.max(self.q_vals) - self.q_vals[act]))

    def train(self, num_episodes):
        """
        imulates running the SMDP algor with the env
        """
        rewards = []   #keep track of total reward so that can use it for results
        budgets = []
        deteriorations = []

        for eps in range(num_episodes):
            self.env.reset()              #reset env at start of each episode
            state = self.env.reset()
            total_reward = 0   
            total_deterioration = 0  
            done = False

            while not done:
                act = self.choose_act()   #simulate taking act

                #need specify the action duration window to use in env step func
                #choose a random num btw 1,5 years to simulate real world randomness
                if self.env.which_algorithm == "SMDP":
                    action_duration = np.random.randint(1, 6)
                else:
                    action_duration = 1

                next_state, reward, done = self.env.step(act, action_duration)  #simulate taking step in env w/ action that has action duration just found
                self.calc_q_val(act, reward, action_duration)                   #simulate updating the q val for taking that action

                deterioration = min(0, np.mean(next_state) - np.mean(state))
                total_deterioration += deterioration 
                total_reward += reward 

            rewards.append(total_reward)
            budgets.append(self.env.budget)
            deteriorations.append(total_deterioration) 

        return rewards, budgets, deteriorations
