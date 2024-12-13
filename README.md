# Team Maintenance Agents - Punna Chowdhurry, Jane Slagle, and Diana Krmzian
# Tufts CS 138 - Final Project


## StructuRL Bridge Maintenance: Leveraging Reinforcement Learning
We aim to explore how three different reinforcement learning algorithms can be applied to bridge infrastructure maintenace. We create en environmemt, InfraPlanner, to represent a bridge and simulate how three different reinforcement learning algorithms perform on maintatining the conditions of the bridge:

(1) Q-learning SMDP

(2) Deep SARSA

(3) [put what Diana's doing here]

## Requirements
- **Python**: 3.11.5
- **Please refer to requirements.txt for the rest of the package requirements**
- **You must run the code in the virtual environment described directly below**

## Getting Started

(1) To get started, download and unzip the code package [input whatever we name our final zip file here]

(2) Access the project folder, [input whatever we call that folder here], via your terminal

(3) Create a virtual environment

```{.py}
python -m venv .venv
```

(4) Activate the environment

```{.py}
source .venv/bin/activate
```

(5) Install the required packages by using the pip command in terminal

```{.py}
 pip install -r requirements.txt
```

## To Run and view results

(1) Open the 'main.py' file
- you may specify how many episodes you want to run the algorithms with, the default is set at 100,000
  
(2) Uncomment the algorithm function you want to see results from in the 'if __name__ == "__main__":' block:
- 'run_each_algor("SMDP", num_episodes)' for running the InfraPlanner environment with an SMDP algorithm
- 'run_each_algor("DeepSARSA", num_episodes)' for running the InfraPlanner environment with a Deep SARSA algorithm

(3) Run 'main.py' to see plotting results and also print statements providing various metrics for each algorithm. The results from each algorithm are clearly labelled if you decide to run all three algorithms at once.

## Code Overview

**InfraPlanner.py**:
a simulation environment for bridge infrastrucutre maintenance on one bridge over a 100 year period. Budget constraints are incoporated with each action having a fixed associated cost. When determining the reward, how well the bridge is improving over time is also considered.

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
The budget is directly tied to each time step. If the agent uses up all of their budget, then a pretty big penalty is applied to the reward, represented as -10.
When there is sufficient budget to take actions, we deduct the cost to take that action from the total budget. Depending on the action taken, the condition of the bridge will either improve or worsen.

- 'do nothing' action: the state (condition) of the bridge worsens by 1% for the time step to simulate how in the real-world neglecting to maintain the bridge will lead to the state of the bridge getting worse over time 
- 'mainteance' action: the state improves by 1% for the time step to simulate how in the real-world, if you reguarly maintain the bridge, it's condition will improve
- 'replace' action: if you replace the bridge entirely then it will reset the bridge to have a perfect condition state

**Reward Function**:
We calculate reward based on if the bridge is improving from previous conditions and also based on how well the agent is maintaining the available budget.
We consider the following factors when calculating reward:

- the current condition of the bridge. We set a threshold that a condition of 80 or above is a bridge in "good" condition. If the bridge is in such a condition, then we positively reward it with +10. We set a threshold that a condition of 20 or below represents a bridge in "poor" condition. If the bridge is in such a condition, we penalize the reward with -10.

- we consider the previous condition compared with the current condition of the bridge to see if it is improving over time. If it is, then we positively reward with +1

- we consider the budget. If the agent surpasses the budget, then we penalize it with -5. If the agent is staying within the bounds of the budget, then we positively reward it with +2
    
**SMDP.py**:
Q-learning based SMDP algorithm representing an agent that is able to interact with the InfraPlanner environment.
It employs an epsilon-greedy policy for action selection and updates Q values by scaling them by the variable action durations since it is SMDP. It handles all training for running through the environment with an SMDP algrorithm.

**DeepSARSA.py**
[add brief description]

**[deep learning file]**
[add description once have it]

**main.py**
