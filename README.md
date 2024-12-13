# Team Maintenance Agents - Punna Chowdhurry, Jane Slagle, and Diana Krmzian
# Tufts CS 138 - Final Project


## StructuRL Bridge Maintenance: Leveraging Reinforcement Learning
[put one sentence description of our model here]

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

 State Space:
 - condition of the bridge represented as an integer value, ranging between 0 and 100 where 100 is a perfect condition and 0 is the worst possible condition
 - the condition of the bridge is always initialized as 40 so that we may observe how our model either improves or worsens the condition based on the actions it chooses to take

Action Space: 
3 possible actions related to bridge infrastructure mainteance tasks:
- do nothing [associated action cost of 0]
- maintenance [associated action cost of 2]
- replace [associated action cost of 5]
The associated cost of each action is taken out of the budget.

[briefly describe step + reward function too to finish this section out]
    
**SMDP.py**:
Q-learning based SMDP algorithm representing an agent that is able to interact with the InfraPlanner environment.
[add little more detail, just in general]

**DeepSARSA.py**
[add brief description]

**[deep learning file]**

**main.py**
