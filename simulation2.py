from env2 import *
import numpy as np
import matplotlib.pyplot as plt
# random seed(2)

NumOfRoads = 10
NumOfPlayers = 200
Algorithm = ['average_sample', 'joint_actions', 'mean_field', 'actor_critic']
np.random.seed(2)
roads = [np.random.rand(3) for _ in range(NumOfRoads)]
np.random.seed(2)
costsHistory = {'average_sample': [], 'joint_actions': [], 'mean_field': [], 'actor_critic': []}
costsMean = {'average_sample': [], 'joint_actions': [], 'mean_field': [], 'actor_critic': []}
roadsChoiceHistory = {'average_sample': [], 'joint_actions': [], 'mean_field': [], 'actor_critic': []}

for algorithm in Algorithm:
    env = SimpleCongestion(NumOfRoads, NumOfPlayers, algorithm, roads)
    for iters in range(500):
        costs, roadsChoice = env.__procudure__()
        costsHistory[algorithm].append(costs)
        # print (roadsChoice)
        # print (np.sum(costs))
        costsMean[algorithm].append(np.sum(costsHistory[algorithm])/(iters+1))
        roadsChoiceHistory[algorithm].append(roadsChoice)

maplet = {'joint_actions': 'JSFP', 'actor_critic': 'ACWFP', 'mean_field': 'MFFP', 'average_sample': 'AS'}

# plt.figure()
# plt.plot(costsHistory)
# plt.title('Game of {} Roads and {} Players with {}'.format(NumOfRoads, NumOfPlayers, maplet[Algorithm]))
# plt.xlabel('Day Number')
# plt.ylabel('Congestion Cost on Each Route')

fig, ax = plt.subplots(1, 4, figsize=(25, 5))
fig.suptitle('The Plots of the Number of Cars on Each Road', fontsize=16, fontweight='bold')
fig.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.1, wspace=.0)
for i in range(len(Algorithm)):
    ax[i].plot(roadsChoiceHistory[Algorithm[i]])
    ax[i].set_title(maplet[Algorithm[i]], loc='center', fontdict={'fontsize': 8, 'fontweight': 'bold'})
    if i > 0:
        plt.setp(ax[i], yticks=[])
ax[0].set_ylabel('Number of Cars on Each Road')
fig.text(0.5, 0.04, 'Day Number', va='center')


plt.figure()
plt.title('The Plot of the Mean of Costs', fontsize=14, fontweight='bold')
plt.xlabel('Day Number')
plt.ylabel('Mean of Costs')
for algorithm in Algorithm:
    plt.plot(costsMean[algorithm], label=maplet[algorithm])
plt.grid()
plt.legend()
plt.show()
