from env2 import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation
import matplotlib.animation as animation
import matplotlib as mpl
from matplotlib.patches import Patch
from scipy.stats import *


maplet = {'joint_actions': 'JSFP', 'actor_critic': 'ACWFP', 'mean_field': 'MFFP', 'average_sample': 'AS'}
NumOfPlayers = 60
np.random.seed(2)
task_indice = np.random.choice(6, NumOfPlayers)
np.random.seed()

################################################################################
Algorithm = ['mean_field', 'joint_actions', 'actor_critic', 'average_sample']
rewardsHistory = [[], [], [], []]
rewardsMean = [[], [], [], []]
Iters = 200
for i in range(len(Algorithm)):
    env = Inventory(NumOfPlayers, Algorithm[i], IfSnippet=False, IfGoal=True, taskIndice=task_indice)
    for iters in range(Iters):
        rewards = env.__procudure__()
        rewardsHistory[i].append(rewards)
        rewardsMean[i].append(np.sum(rewardsHistory[i])/(iters+1))
plt.figure(figsize=(8,6))
plt.plot(rewardsMean[0], label='MFFP')
plt.plot(rewardsMean[1], label='JSFP')
plt.plot(rewardsMean[2], label='ACWFP')
plt.plot(rewardsMean[3], label='AS')
plt.title('The Plot of the Mean of Rewards in One Situation')
plt.xlabel('t')
plt.ylabel('Mean of Reward Values')
plt.grid()
plt.legend()
plt.show()
################################################################################
# Algorithm = 'actor_critic'
# # Iters = 50
# Iters = 1000
# Steps = 1000
# env = Inventory(NumOfPlayers, Algorithm, IfSnippet=False, IfGoal=True, taskIndice=task_indice)
# frames = []
# frames.append(env.layout_)
# print (frames[-1])
# for step in range(Steps):
#     print ('Step {}'.format(step))
#     last_rewards = np.zeros((NumOfPlayers, ))
#     env.reset_layout()
#     for iters in range(Iters):
#         if iters == Iters - 1:
#             env.record_flag = True
#         rewards = env.__procudure__()
#         env.record_flag = False
#         if np.sum(np.abs(np.array(rewards) - last_rewards)) < 1e-1:
#             break
#         last_rewards = np.array(rewards)
#     env.update_layout()
#     frames.append(env.layout_)
#     env.update_start_states(env.states)
#     num_finish = sum(env.check_assignments())
#     print (frames[-1])
#     print ('The number of successful assignments is {}.'.format(num_finish))
#     print ('The number of wall crash is {}.'.format(env.wall_crash))
#     print ('The number of conflicts is {}.\n'.format(env.conflicts))
#     if num_finish == 60:
#         break
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
#     if t == len(frames):
#         t = 0
#     ax.imshow(frames[t], interpolation='nearest', vmin=-1, vmax=2, cmap='tab20b')
#     t += 1
# ani = FuncAnimation(fig, update, frames=len(frames), interval=10)
# ani.save(maplet[Algorithm]+'_dynamics'+str(NumOfPlayers)+'.mp4', writer=writer)
################################################################################
# Algorithm = ['mean_field', 'joint_actions']
# Iters = 10000
# Steps = 1000
# experiments_times = 100
# experiments = [{'num_finish': [], 'wall_crash': [], 'conflicts': [], 'iters': []},\
#                 {'num_finish': [], 'wall_crash': [], 'conflicts': [], 'iters': []}]
# for i in range(len(Algorithm)):
#     for exp in range(experiments_times):
#         env = Inventory(NumOfPlayers, Algorithm[i], IfSnippet=False, IfGoal=True, taskIndice=task_indice)
#         frames = []
#         frames.append(env.layout_)
#         print ('This is the experiment {}!'.format(exp))
#         print(frames[-1])
#         for step in range(Steps):
#             print ('Step {}'.format(step))
#             last_rewards = np.zeros((NumOfPlayers, ))
#             env.reset_layout()
#             for iters in range(Iters):
#                 if iters == Iters - 1:
#                     env.record_flag = True
#                 rewards = env.__procudure__()
#                 env.record_flag = False
#                 if np.sum(np.abs(np.array(rewards) - last_rewards)) < 1e-4:
#                     experiments['iters'].append(iters)
#                     break
#                 last_rewards = np.array(rewards)
#             env.update_layout()
#             frames.append(env.layout_)
#             env.update_start_states(env.states)
#             num_finish = sum(env.check_assignments())
#             print (frames[-1])
#             print ('The number of successful assignments is {}.'.format(num_finish))
#             print ('The number of wall crash is {}.'.format(env.wall_crash))
#             print ('The number of conflicts is {}.\n'.format(env.conflicts))
#             if num_finish == 60:
#                 experiments[i]['num_finish'].append(step)
#                 experiments[i]['wall_crash'].append(env.wall_crash)
#                 experiments[i]['conflicts'].append(env.conflicts)
#                 break
# np.save('./mffp_exp3.npy', np.array([experiments[0]['num_finish'], experiments[0]['wall_crash'], experiments[0]['conflicts'], experiments[0]['iters']]))
# np.save('./jsfp_exp3.npy', np.array([experiments[1]['num_finish'], experiments[1]['wall_crash'], experiments[1]['conflicts'], experiments[1]['iters']]))
################################################################################
# for name in experiments[0].keys():
#     twosample_results = ttest_ind(experiments[0][name], experiments[1][name])
#     print ('This is the results of {}: {}.'.format(name, twosample_results))
