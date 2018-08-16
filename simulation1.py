from env2 import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# random seed(2)

Algorithm = 'average_sample'

env = StrategyGame(Algorithm)
actionsHistory = [[], []]

for iters in range(int(400)):
    actions = env.__procudure__()
    actionsHistory[0].append(actions[0])
    actionsHistory[1].append(actions[1])
    print (actions)

maplet = {'joint_actions': 'JSFP', 'actor_critic': 'ACWFP', 'mean_field': 'MFFP', 'average_sample': 'AS'}

plt.figure()
plt.plot(actionsHistory[0], '.')
plt.title('Strategy Choice of Player {} with {}'.format(1, maplet[Algorithm]))
plt.xlabel('t')
plt.ylabel('action')
plt.figure()
plt.plot(actionsHistory[1], '.')
plt.title('Strategy Choice of Player {} with {}'.format(2, maplet[Algorithm]))
plt.xlabel('t')
plt.ylabel('action')
plt.show()
