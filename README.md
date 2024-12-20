# StructuRL Bridge Maintenance: Leveraging Reinforcement Learning 
## Punna Chowdhurry, Jane Slagle, and Diana Krmzian
## Tufts CS 138 - Final Project

## Overview

Please note that we took inspiration from https://github.com/CivML-PolyMtl/InfrastructuresPlanner for this project.

We aim to explore how three different reinforcement learning algorithms can be applied to bridge infrastructure maintenace. We create a custom environmemt, an extremely simplified form inspired from the environment provided by https://github.com/CivML-PolyMtl/InfrastructuresPlanner, InfraPlanner, to represent a bridge and simulate how three different reinforcement learning algorithms perform on maintatining the conditions of the bridge:

(1) Q-learning SMDP (Semi Markov Decision Process)

(2) Deep SARSA

(3) Deep Q-Learning

## Requirements
- **python**: 3.11.5
- **numpy**: 1.22.0
- **matplotlib**: 3.6.3

## To Run and View Results

(1) Open the 'main.py' file
- you may specify how many episodes you want to run the algorithms with, the default is set at 10,000
  
(2) Uncomment the algorithm function you want to see results from in the 'if __name__ == "__main__":' block:
- 'run_each_algor("SMDP", num_episodes)' for running the InfraPlanner environment with an SMDP algorithm
- 'run_each_algor("DeepSARSA", num_episodes)' for running the InfraPlanner environment with a Deep SARSA algorithm
- 'run_each_algor("DeepQLearning", num_episodes)' for running the InfraPlanner environment with a Deep Q-Learning algorithm

(3) Run 'main.py' to see plotting results and also print statements providing various metrics for each algorithm. The results from each algorithm are clearly labelled if you decide to run all three algorithms at once

## Code Overview

All of the code for our project is hosted on the following GitHub repo: https://github.com/janeslagle/CS_138_final_project/tree/main

**InfraPlanner.py**:
a simulation environment for bridge infrastrucutre maintenance on one bridge over a 100 year period. Budget constraints are incorporated with each action having a fixed associated cost. When determining the reward, how well the condition of the bridge is improving over time as well as how well the budget is being managed are both considered.

 **State Space**:
 - condition of the bridge represented as an integer value, ranging between 0 and 100 where 100 is a perfect condition and 0 is the worst possible condition
 - the condition of the bridge is always initialized as 40 so that we may observe how our model either improves or worsens the condition based on the actions it chooses to take

**Action Space**: 
3 possible actions related to bridge infrastructure mainteance tasks:
- do nothing [associated action cost of 0]
- maintenance [associated action cost of 2]
- replace [associated action cost of 5]
  
The associated cost of each action is taken out of the budget.

**Step Function**:
If running the SMDP algorithm, we employ variable time step lengths for each action. For non-SMDP algorithms, each action takes one time step as per usual.
The budget is directly tied to each time step. If the agent uses up all of the available budget, then a pretty big penalty is applied to the reward, represented as -10.
When there is sufficient budget to take actions, we deduct the cost to take that action from the total budget. Depending on the action taken, the condition of the bridge will either improve or worsen.

- 'do nothing' action: the state (condition) of the bridge worsens by 1% for the time step to simulate how in the real-world neglecting to maintain the bridge will lead to the state of the bridge getting worse over time 
- 'mainteance' action: the state improves by 1% for the time step to simulate how in the real-world, if you reguarly maintain the bridge, it's condition will improve over time
- 'replace' action: if you replace the bridge entirely then it will reset the bridge to have a perfect condition state of 100

**Reward Function**:
We calculate reward based on if the bridge is improving from previous conditions and also based on how well the agent is maintaining the available budget.
We consider the following factors when calculating reward:

- the current condition of the bridge. We set a threshold that a condition of 80 or above is a bridge in "good" condition. If the bridge is in such a condition, then we positively reward it with +10. We set a threshold that a condition of 20 or below represents a bridge in "poor" condition. If the bridge is in such a condition, we penalize the reward with -10.

- we consider the previous condition compared with the current condition of the bridge to see if it is improving over time. If it is, then we positively reward with +3

- we consider the budget. If the agent surpasses the budget, then we penalize it with -5. If the agent is staying within the bounds of the budget, then we positively reward it with +2
    
**SMDP.py**:
Q-learning based SMDP algorithm representing an agent that is able to interact with the InfraPlanner environment.
It employs an epsilon-greedy policy for action selection and updates Q values by scaling them by the variable action durations since it is SMDP. It handles all training for running through the environment with an SMDP algorithm.

**DeepSARSA.py**:
Implements a Deep SARSA agent that uses a neural network to approximate the Q values. It uses an epsilon-greedy approach for action selection and also incorporates weights. This integration of deep learning allows the agent to handle complex decision-making tasks effectively.
- Related works: 
    - https://arxiv.org/pdf/1702.03118 
        - This research uses Deep SARSA (λ) agent with SiLU and dSiLU hidden units.
    - https://medium.com/swlh/learning-with-deep-sarsa-openai-gym-c9a470d027a 
        - Implements Deep SARSA using OpenAI Gym’s Cartpole environment and Keras-RL. 
        - In most episodes, the agent managed to maintain rewards within a range of 140–160 steps. 
    - https://github.com/JohDonald/Deep-Q-Learning-Deep-SARSA-LunarLander-v2 
        - This repo discusses implementing and comparing Deep Q-Learning (DQN) and Deep SARSA for solving the LunarLander-v2 environment from OpenAI Gym.

**DeepQLearning.py**:
We use a basic Deep Q-Learning Algorithm (DQL) which is based on a simple neural network to help us make strong predictions on best actions to take in each situation. The algorithm uses a straightforward method to update its decisions over time, learning from the rewards it gets for each of the different actions. It also tries a mix of exploring new actions and sticking to what it already knows that works well(using an epsilon-greedy approach). This version helps us keep things simple, but focuses more on the basics to help the agent learn in an effective way in the InfraPlanner environment.

**main.py**:
By following the steps outlined in the "To Run and View Results" section above, we can run 'main.py' to obtain results for each of our three algorithms. For each algorithm the following results are obtained:

- a plot showing (1) the culmulative rewards over all episodes and (2) the average reward per episode for all episodes (both the original and smoothed version)
- a plot showing (1) the remaining budget for all episodes and (2) the deteoriation over episodes
- the initial state condition of the bridge
- the initial budget of the bridge
- the average culmulative reward
- the percentage of times each action out of the three possible actions was taken out of the total number of actions taken
- the cost efficiency (ratio of the average cumulative reward to the total cost spent)
- the final environment state with the percentage of how much the agent either improved or deteroriated the bridge
- the final budget with the percentage of the total budget used and the percentage of the total budget leftover
