# Team Maintenance Agents - Punna Chowdhurry, Jane Slagle, and Diana Krmzian
# Tufts CS 138 - Final Project
# InfraPlanner.py

import numpy as np

class InfraPlanner:
    def __init__(self, total_years=100, which_algorithm="SMDP"):
        self.total_years = total_years
        self.current_year = 0
        self.max_budget = 100
        self.budget = self.max_budget
        self.state_size = 1
        self.state = np.ones(self.state_size) * 40
        self.actions = ['do nothing', 'maintenance', 'replace']
        self.action_costs = {'do nothing': 0, 'maintenance': 2, 'replace': 5} 
        self.which_algorithm = which_algorithm
    
    def step(self, action, action_duration=1):
        reward = 0
        done = False       #keeps track of if episode is finished or not

        #if running env w/ SMDP algorithm then use action_duration to det how long action should take
        #with non-SMDP algorithms, each action should only take 1 time step
        if self.which_algorithm == "SMDP":
            time_step_length = action_duration
        else:
            time_step_length = 1

        #increment current yr based on the time increment have for the action (will increment by 1 year for non-SMDP and also increment
        #for however many years the SMDP actions end up taking)
        self.current_year += time_step_length

        #store copy of current state (bridge's current condt) so that can give reward if improved upon it's condt from last time
        prev_condition = self.state.copy() 

        #episode finished when have gone throuhg total_years
        if self.current_year >= self.total_years:
            done = True  

        #calc how much of the budget was used from taking that action for the amount of time the action took
        action_cost = self.action_costs[action] * time_step_length

        #penalize if go over budget
        #if not enough budget is left to take the inputted action then penalize it with reward
        #and can't take the action so state doesn't change
        if self.budget < action_cost:
            reward -= 10 
            next_state = self.state
        else:
            self.budget -= action_cost   #otherwise if can take action then get the budget left for next time step

            # now update state based on action taking. clip each to make sure the bridge condition stays within the bounds of 0-100
            if action == 'do nothing':
                #if bridge is being neglected, deteroriate its condition by 1%, scale it by the time step length for SMDP
                next_state = np.clip(self.state * (0.99 ** time_step_length), 0, 100)
            elif action == 'maintenance':
                #if bridge is being maintained, improve its condition by 1%
                next_state = np.clip(self.state * (1.01 ** time_step_length), 0, 100)
            elif action == 'replace':
                #have replaced the bridge so its in perfect condition
                next_state = self.state * 100

            #calc reward based on prev state + new state + scale it by the action duration
            reward += self.calculate_reward(next_state, prev_condition) * time_step_length
            
        self.state = next_state  #update state
	
        return next_state, reward, done
    
    def calculate_reward(self, condition, prev_condition):
        reward = 0

        #a bridge with state greater than or equal to 80 is a bridge in "good" condition, increase the reward
        #take the mean in case state_size is greater than 1 then can account for all
        if condition >= 80:
            reward += 10
        
        #a bridge with state less than or equal to 20 is in "bad" condition, so penalize the reward
        elif condition <= 20:
            reward -= 10

        #use prev_condition input to add additional reward if the condt is improving!
        if condition > prev_condition:
            reward += 3    #reward for making progress

        #checks if agent overspent budget: if did then penalize it. otherwise reward it for having good budget management skills!
        if self.budget < 0:
            reward -= 5
        else:
            reward += 2

        return reward

    def reset(self):
        #re-initialize everything!
        self.current_year = 0
        self.budget = self.max_budget
        self.state = np.ones(self.state_size) * 40

        return self.state
