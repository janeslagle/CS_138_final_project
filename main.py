# Team Maintenance Agents - Punna Chowdhurry, Jane Slagle, and Diana Krmzian
# Tufts CS 138 - Final Project
# main.py

import numpy as np
import matplotlib.pyplot as plt
from InfraPlanner import InfraPlanner
from SMDP import SMDP
from DeepSARSA import DeepSARSA
from DeepQLearning import DeepQLearning

def plot_rewards(rewards, budgets, deteriorations, which_algorithm):
    #get a moving average (rolling mean) of the rewards to smooth the rewards over episodes curve
    window_size = 100 

    if len(rewards) >= window_size:
        smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
    else:
        smoothed_rewards = rewards

    plt.subplot(2, 1, 1)
    plt.plot(np.cumsum(rewards), label='Cumulative Rewards', color = "deeppink")
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Rewards')
    plt.title(str(which_algorithm) + ' Cumulative Rewards Over Episodes')

    plt.subplot(2, 1, 2)
    plt.plot(rewards, label='Rewards per Episode', color='lightskyblue', alpha=0.5)

    if len(smoothed_rewards) < len(rewards):
        smoothed_x = np.arange(window_size - 1, len(rewards))
    else:
        smoothed_x = np.arange(len(smoothed_rewards))

    plt.plot(smoothed_x, smoothed_rewards, label=f'Mean (window={window_size})', color='cornflowerblue', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(str(which_algorithm) + ' Rewards Per Episode (Original Results & Smoothed Average)')
    plt.legend()

    plt.suptitle("Culmulative and Per-Epsiode Rewards for " + str(which_algorithm))
    plt.tight_layout()
    plt.show()

    plt.subplot(2, 1, 1)
    plt.plot(budgets, label="Remaining Budget", color='seagreen')
    plt.xlabel("Episodes")
    plt.ylabel("Budget")
    plt.title(str(which_algorithm) + " Remaining Budget Over Episodes")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(deteriorations, label='Deterioration Per Episode', color='tomato')
    plt.xlabel('Episodes')
    plt.ylabel('Deterioration')
    plt.title(f'{which_algorithm} Deterioration Over Episodes')
    plt.legend()

    plt.suptitle("Remaining Budget & Deterorioration Per Episode for " + str(which_algorithm))
    plt.tight_layout()
    plt.show()

def agent_results(env, agent, rewards):
    avg_reward = np.round(np.mean(rewards), 2)
    print("Average Cumulative Reward: " + str(avg_reward))

    #store amt times took each act as a dict where key = the action, value = amount of times took that action in agent when running algor
    #each_act_amt = {action: agent.all_acts_taken.count(action) for action in env.actions}
    each_act_amt = {action: agent.all_acts_taken.count(action) for action in env.actions}

    total_acts_taken = sum(each_act_amt.values())

    print("Percentage of times took each action out of " + str(total_acts_taken) + " total times:")

    for act, num_times in each_act_amt.items():
        act_percent = (num_times / total_acts_taken) * 100
        print("Action " + str(act) + " " + str(np.round(act_percent, 2)) + "% " + "(" + str(num_times) + " times)")
    
    total_cost_spent = env.max_budget - env.budget

    #got some runtime divide by 0s errors so accounting for that here
    if total_cost_spent > 0:
        cost_effic = avg_reward / total_cost_spent
    else:
        cost_effic = 0

    print("Total Cost Efficiency: " + str(np.round(cost_effic, 2)))

def run_each_algor(which_algorithm, num_episodes):
    print(str(which_algorithm) + " results:")
    print("")

    #set up env
    env = InfraPlanner(which_algorithm=which_algorithm)

    initial_condt = int(env.reset())
    initial_budget = env.budget

    print("Initial environment state (bridge condition):", initial_condt)
    print("Initial budget:", initial_budget)

    #now set up agent depending on which algor we are running 
    if which_algorithm == "SMDP":
        agent = SMDP(env)
    elif which_algorithm == "DeepSARSA":
        agent = DeepSARSA(env)
    elif which_algorithm == "DeepQLearning":
        agent = DeepQLearning(env)

    #train agent on all episodes
    rewards, budgets, deteriorations = agent.train(num_episodes=num_episodes)

    #get all results out from training on env w/ algor
    plot_rewards(rewards, budgets, deteriorations, which_algorithm=which_algorithm)

    #now get all wanted performance metrics out
    agent_results(env, agent, rewards)

    #print out final results for condt + budget w/ how much each of them changed
    final_condt = int(env.state)
    print("Final environment state:", final_condt)

    condt_percent = np.round(((final_condt - initial_condt) / (100 - initial_condt)) * 100, 2)

    #figure out if condition of bridge improved or deteroriated
    if condt_percent > 0:
        print("Bridge Condition improved by " + str(condt_percent) + "%")
    elif condt_percent < 0:
        print("Bridge condition deteriorated by " + str(condt_percent * -1) + "%")
    else: 
        print("Bridge condition did not change.")

    final_budget = env.budget
    print("Final budget:", final_budget)

    spent_budget = ((initial_budget - final_budget) / initial_budget) * 100
    budget_left = 100 - spent_budget

    print("Percentage of total budget spent: " + str(spent_budget) + "%")
    print("Percentage of total budget leftover: " + str(budget_left) + "%")

if __name__ == '__main__':
    num_episodes = 10000   #want run all algor w/ the same number of episodes = 10000
    
    run_each_algor("SMDP", num_episodes)

    print("")
    print("")

    run_each_algor("DeepSARSA", num_episodes)

    print("")
    print("")

    run_each_algor("DeepQLearning", num_episodes)

    pass
