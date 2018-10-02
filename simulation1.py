from env import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

Algorithm = ['sample_average', 'joint_actions', 'mean_field', 'actor_critic']
actionsHistoryx = {'sample_average': [], 'joint_actions': [], 'mean_field': [], 'actor_critic': []}
actionsHistoryy = {'sample_average': [], 'joint_actions': [], 'mean_field': [], 'actor_critic': []}
np.random.seed()

for algorithm in Algorithm:
    env = StrategyGame(algorithm)
    for iters in range(int(400)):
        actions = env.__procudure__()
        actionsHistoryx[algorithm].append(actions[0])
        actionsHistoryy[algorithm].append(actions[1])
        print (actions)

maplet = {'joint_actions': 'JSFPI', 'actor_critic': 'ACGWFP', 'mean_field': 'MFFP', 'sample_average': 'SA'}

fig, ax = plt.subplots(2, 4, figsize=(30, 5))
# fig.suptitle('The Plots of the Solutions', fontsize=16, fontweight='bold')
fig.subplots_adjust(left=0.045, right=0.98, top=0.9, bottom=0.1, wspace=.0, hspace=0.4)
for i in range(len(Algorithm)):
    ax[0][i].plot(actionsHistoryx[Algorithm[i]], '*')
    ax[1][i].plot(actionsHistoryy[Algorithm[i]], '+')
    ax[0][i].set_ylim([-0.5, 1.5])
    ax[1][i].set_ylim([-0.5, 1.5])
    ax[0][i].set_xlim([-10, 450])
    ax[1][i].set_xlim([-10, 450])
    ax[0][i].set_title(maplet[Algorithm[i]], loc='center', fontdict={'fontsize': 18, 'fontweight': 'bold'})
    ax[1][i].set_title(maplet[Algorithm[i]], loc='center', fontdict={'fontsize': 18, 'fontweight': 'bold'})
    ax[0][i].tick_params(labelsize=16)
    ax[1][i].tick_params(labelsize=16)
    if i > 0:
        plt.setp(ax[0][i], yticks=[])
        plt.setp(ax[1][i], yticks=[])

ax[0][0].set_ylabel('Action of Player 1', fontsize=18)
ax[1][0].set_ylabel('Action of Player 2', fontsize=18)
fig.text(0.5, 0.018, 'Iters', va='center', fontsize=18)

plt.show()
