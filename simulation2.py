from env2 import *
import numpy as np
import matplotlib.pyplot as plt
# random seed(2)

NumOfRoads = 10
NumOfPlayers = 200
Algorithm = ['sample_average', 'joint_actions', 'mean_field', 'actor_critic']
np.random.seed(2)
roads = [np.random.rand(3) for _ in range(NumOfRoads)]
np.random.seed(2)
costsHistory = {'sample_average': [], 'joint_actions': [], 'mean_field': [], 'actor_critic': []}
costsMean = {'sample_average': [], 'joint_actions': [], 'mean_field': [], 'actor_critic': []}
roadsChoiceHistory = {'sample_average': [], 'joint_actions': [], 'mean_field': [], 'actor_critic': []}

for algorithm in Algorithm:
    env = SimpleCongestion(NumOfRoads, NumOfPlayers, algorithm, roads)
    for iters in range(500):
        costs, roadsChoice = env.__procudure__()
        costsHistory[algorithm].append(costs)
        costsMean[algorithm].append(np.sum(costsHistory[algorithm])/(iters+1))
        roadsChoiceHistory[algorithm].append(roadsChoice)

maplet = {'joint_actions': 'JSFPI', 'actor_critic': 'ACGWFP', 'mean_field': 'MFFP', 'sample_average': 'SA'}

fig, ax = plt.subplots(1, 4, figsize=(30, 3))
fig.subplots_adjust(left=0.046, right=1., top=.90, bottom=0.18, wspace=.0)
for i in range(len(Algorithm)):
    ax[i].plot(roadsChoiceHistory[Algorithm[i]], linewidth=2)
    ax[i].set_title(maplet[Algorithm[i]], loc='center', fontdict={'fontsize': 16, 'fontweight': 'bold'})
    ax[i].tick_params(labelsize=16)
    ax[i].set_xlim([0, 550])
    ax[i].set_ylim([-5, 60])
    if i > 0:
        plt.setp(ax[i], yticks=[])
ax[0].set_ylabel('Number of Cars on Each Road', fontsize=13)
fig.text(0.5, 0.04, 'Day Number', va='center', fontsize=13)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
fig.subplots_adjust(left=0.15, right=.97, top=0.93, bottom=0.16, wspace=.0)
ax.set_xlabel('Day Number', fontsize=18)
ax.set_ylabel('Mean of Costs', fontsize=18)
for algorithm in Algorithm:
    plt.plot(costsMean[algorithm], label=maplet[algorithm], linewidth=2.0)
ax.set_xlim([0, 500])
ax.tick_params(labelsize=16)
plt.grid()
ax.legend(fontsize=16)
plt.show()
