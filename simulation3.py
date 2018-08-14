from env2 import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation
import matplotlib.animation as animation
import matplotlib as mpl
from matplotlib.patches import Patch
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import pandas as pd
from scipy.stats import *

np.random.seed(2)

################################################################################
# NumOfPlayers = 60
# Algorithm = 'joint_actions'
# Iters = 60
# Steps = 200
# maplet = {'joint_actions': 'JSFP', 'actor_critic': 'ACWFP', 'mean_field': 'MFFP'}
# task_indice = np.random.choice(6, NumOfPlayers)
# env = Inventory(NumOfPlayers, Algorithm, IfSnippet=False, IfGoal=True, taskIndice=task_indice)
# rewardsHistory = []
# rewardsMean = []
# env.plot_grid()
# for iters in range(Iters):
#     rewards = env.__procudure__(iters)
#     if iters == (Iters-1):
#         env.plot_grid()
#     rewardsHistory.append(rewards)
#     rewardsMean.append(np.sum(rewardsHistory)/(iters+1))
# plt.figure()
# plt.plot(rewardsMean)
# plt.title('Game of {} Players with {}'.format(NumOfPlayers, maplet[Algorithm]))
# plt.xlabel('t')
# plt.ylabel('Mean of Total Rewards')
################################################################################
# NumOfPlayers = 60
# Algorithm = 'joint_actions'
# Iters = 60
# Steps = 200
# maplet = {'joint_actions': 'JSFP', 'actor_critic': 'ACWFP', 'mean_field': 'MFFP'}
# task_indice = np.random.choice(6, NumOfPlayers)
# env = Inventory(NumOfPlayers, Algorithm, IfSnippet=False, IfGoal=True, taskIndice=task_indice)
# frames = []
# frames.append(env.layout_)
# for step in range(Steps):
#     print ('Step {}'.format(step))
#     print (frames[step])
#     for iters in range(Iters):
#         if iters == Iters - 1:
#             env.record_flag = True
#         rewards = env.__procudure__(iters)
#         env.record_flag = False
#     frames.append(env.layout_)
#     env.update_start_states(env.states)
#     num_finish = sum([1 for tick in env.check_assignments() if tick])
#     print ('The number of successful assignments is {}.'.format(num_finish))
#     print ('The number of wall crash is {}.'.format(env.wall_crash))
#     print ('The number of conflicts is {}.\n'.format(env.conflicts))
# frames = frames[-1:]+frames
# del frames[-1:]
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=4, metadata=dict(artist='Jianhong_Wang'), bitrate=1800)
# fig, ax = plt.subplots(figsize=(14, 8))
# plt.xticks([])
# plt.yticks([])
# plt.title('Inventory Game of {} Players with {}'.format(NumOfPlayers, maplet[Algorithm]))
# cmap = plt.cm.get_cmap('tab20b', 4).colors
# legend_elements = [Patch(facecolor=cmap[0], edgecolor='k', label='Wall'),\
#                    Patch(facecolor=cmap[1], edgecolor='k', label='Vacant Position'),\
#                    Patch(facecolor=cmap[2], edgecolor='k', label='Position with 1 Agent'),\
#                    Patch(facecolor=cmap[3], edgecolor='k', label='Storage Rack')]
# plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# t = 0
# def update(*args):
#     global t
#     if t == Steps:
#         t = 0
#     ax.imshow(frames[t], interpolation='nearest', vmin=-1, vmax=2, cmap='tab20b')
#     t += 1
# ani = FuncAnimation(fig, update, frames=Steps, interval=10)
# ani.save(maplet[Algorithm]+'_dynamics'+str(NumOfPlayers)+'.mp4', writer=writer)
################################################################################
NumOfPlayers = 60
Algorithm = ['mean_field', 'joint_actions']
Iters = 60
Steps = 70
maplet = {'joint_actions': 'JSFP', 'actor_critic': 'ACWFP', 'mean_field': 'MFFP'}
experiments_times = 50
experiments = [{'num_finish': [], 'wall_crash': [], 'conflicts': []}, {'num_finish': [], 'wall_crash': [], 'conflicts': []}]
task_indice = np.random.choice(6, NumOfPlayers)
for i in range(len(Algorithm)):
    for exp in range(experiments_times):
        env = Inventory(NumOfPlayers, Algorithm[i], IfSnippet=False, IfGoal=True, taskIndice=task_indice)
        frames = []
        frames.append(env.layout_)
        print ('This is the experiment {}!'.format(exp))
        print(frames[-1])
        for step in range(Steps):
            print ('Step {}'.format(step))
            env.reset_layout()
            for iters in range(Iters):
                if iters == Iters - 1:
                    env.record_flag = True
                rewards = env.__procudure__()
                env.record_flag = False
            env.update_layout()
            frames.append(env.layout_)
            env.update_start_states(env.states)
            num_finish = sum(env.check_assignments())
            print (frames[-1])
            print ('The number of successful assignments is {}.'.format(num_finish))
            print ('The number of wall crash is {}.'.format(env.wall_crash))
            print ('The number of conflicts is {}.\n'.format(env.conflicts))
            if num_finish == 60:
                experiments[i]['num_finish'].append(step)
                experiments[i]['wall_crash'].append(env.wall_crash)
                experiments[i]['conflicts'].append(env.conflicts)
                break
################################################################################
# plt.show()
# true_mus = {'num_finish': 50, 'wall_crash': 4, 'conflicts': 2}
for name in experiments[0].keys():
    twosample_results = ttest_ind(experiments[0][name], experiments[1][name])
    print ('This is the results of {}: {}.'.format(name, twosample_results))
# matrix_onesample = [
#     ['', 'Test Statistic', 'p-value'],
#     ['Sample Data', onesample_results[0], onesample_results[1]]
# ]
#
# onesample_table = FF.create_table(matrix_onesample, index=True)
# py.iplot(onesample_table, filename='onesample-table')
