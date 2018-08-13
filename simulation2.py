from env2 import *
import numpy as np
import matplotlib.pyplot as plt
# random seed(2)

np.random.seed(2)

NumOfRoads = 10
NumOfPlayers = 100
Algorithm = 'joint_actions'
roads = [np.random.rand(3) for _ in range(NumOfRoads)]
env = SimpleCongestion(NumOfRoads, NumOfPlayers, Algorithm, roads)
costsHistory = []
costsMean = []
roadsChoiceHistory = []

for iters in range(200):
    costs, roadsChoice = env.__procudure__()
    costsHistory.append(costs)
    print (roadsChoice)
    print (np.sum(costs))
    costsMean.append(np.sum(costsHistory)/(iters+1))
    roadsChoiceHistory.append(roadsChoice)

maplet = {'joint_actions': 'JSFP', 'actor_critic': 'ACWFP', 'mean_field': 'MFFP'}

plt.figure()
plt.plot(costsHistory)
plt.title('Game of {} Roads and {} Players with {}'.format(NumOfRoads, NumOfPlayers, maplet[Algorithm]))
plt.xlabel('Day Number')
plt.ylabel('Congestion Cost on Each Route')
plt.figure()
plt.plot(roadsChoiceHistory)
plt.title('Game of {} Roads and {} Players with {}'.format(NumOfRoads, NumOfPlayers, maplet[Algorithm]))
plt.xlabel('Day Number')
plt.ylabel('Number of Drivers on Each Route')
plt.figure()
plt.plot(costsMean)
plt.title('Game of {} Roads and {} Players with {}'.format(NumOfRoads, NumOfPlayers, maplet[Algorithm]))
plt.xlabel('Day Number')
plt.ylabel('Mean of Total Costs')
plt.show()
