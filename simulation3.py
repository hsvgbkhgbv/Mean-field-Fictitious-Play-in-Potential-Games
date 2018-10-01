from env2 import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation
import matplotlib.animation as animation
import matplotlib as mpl
from matplotlib.patches import Patch
import argparse
np.random.seed(2)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=int, default=0)
parser.add_argument('--algo', type=int, default=0)
args = parser.parse_args()

maplet = {'joint_actions': 'JSFPI', 'actor_critic': 'ACGWFP', 'mean_field': 'MFFP', 'sample_average': 'SA'}
NumOfPlayers = 60
task_indice = np.random.choice(6, NumOfPlayers)

if args.mode == 0:
    Algorithm = ['mean_field', 'joint_actions', 'actor_critic', 'sample_average']
    rewardsHistory = [[], [], [], []]
    rewardsMean = [[], [], [], []]
    Iters = 1000
    for i in range(len(Algorithm)):
        env = Inventory(NumOfPlayers, Algorithm[i], IfSnippet=False, IfGoal=True, taskIndice=task_indice)
        for iters in range(Iters):
            rewards = env.__procudure__()
            rewardsHistory[i].append(rewards)
            rewardsMean[i].append(np.sum(rewardsHistory[i])/(iters+1))
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    fig.subplots_adjust(left=0.13, right=.95, top=0.95, bottom=0.13, wspace=.0)
    ax.plot(rewardsMean[0], label='MFFP', linewidth=2)
    ax.plot(rewardsMean[1], label='JSFPI', linewidth=2)
    ax.plot(rewardsMean[2], label='ACGWFP', linewidth=2)
    ax.plot(rewardsMean[3], label='SA', linewidth=2)
    fig.text(0.5, 0.018, 'Iters', va='center', fontsize=18)
    ax.set_ylabel('Mean of Rewards', fontsize=18)
    plt.grid()
    ax.legend(fontsize=16)
    ax.set_xlim([0, 1000])
    ax.tick_params(labelsize=16)
    plt.show()
elif args.mode == 1:
    mapping = {0: 'joint_actions', 1: 'actor_critic', 2: 'mean_field', 3: 'sample_average'}
    Algorithm = mapping[args.algo]
    Iters = 1000
    Steps = 1000
    conflicts = 0
    wall_crash = 0
    env = Inventory(NumOfPlayers, Algorithm, IfSnippet=False, IfGoal=True, taskIndice=task_indice)
    frames = []
    frames.append(env.layout_)
    print (frames[-1])
    for step in range(Steps):
        print ('Step {}'.format(step))
        last_mean_rewards = 0
        env.reset_layout()
        for iters in range(Iters):
            rewards = env.__procudure__()
            mean_rewards = last_mean_rewards + 1/(iters+1) * (np.sum(rewards) - last_mean_rewards)
            if np.abs(mean_rewards - last_mean_rewards) < .1:
                print ('This is the iters num: {}'.format(iters+1))
                break
            elif iters == Iters - 1:
                print ('This is the iters num: {}'.format(iters+1))
            else:
                env.conflicts = 0
                env.wall_crash = 0
            last_mean_rewards = mean_rewards
        conflicts += env.conflicts
        wall_crash += env.wall_crash
        env.update_layout()
        print (env.layout_)
        frames.append(env.layout_)
        env.update_start_states(env.states)
        num_finish = sum(env.check_assignments())
        print ('It is the stats of states: \n {}'.format(env.states))
        print ('The number of successful assignments is {}.'.format(num_finish))
        print ('The number of wall crash is {}.'.format(wall_crash))
        print ('The number of conflicts is {}.'.format(conflicts))
        if num_finish == 60:
            break
    frames = frames[-1:]+frames
    del frames[-1:]
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=4, metadata=dict(artist='Jianhong_Wang'), bitrate=1800)
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.xticks([])
    plt.yticks([])
    plt.title('Inventory Game of {} Players with {}'.format(NumOfPlayers, maplet[Algorithm]))
    cmap = plt.cm.get_cmap('Accent', 5).colors
    legend_elements = [Patch(facecolor=cmap[0], edgecolor='k', label='Wall'),\
                       Patch(facecolor=cmap[1], edgecolor='k', label='Vacant Position'),\
                       Patch(facecolor=cmap[2], edgecolor='k', label='Position with 1 Agent'),\
                       Patch(facecolor=cmap[3], edgecolor='k', label='Position with Conflicts'),\
                       Patch(facecolor=cmap[4], edgecolor='k', label='Storage Rack')]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    t = 0
    def update(*args):
        global t
        if t == len(frames):
            t = 0
        ax.imshow(frames[t], interpolation='nearest', vmin=-1, vmax=3, cmap=plt.cm.get_cmap('Accent', 5))
        t += 1
    ani = FuncAnimation(fig, update, frames=len(frames), interval=10)
    ani.save(maplet[Algorithm]+'_dynamics'+str(NumOfPlayers)+'.mp4', writer=writer)
elif args.mode == 2:
    Algorithm = ['mean_field', 'joint_actions']
    Iters = 1000
    Steps = 1000
    conflicts = 0
    wall_crash = 0
    experiments_times = 100
    experiments = [{'num_finish': [], 'wall_crash': [], 'conflicts': [], 'iters': [], 'rewards': []},\
                    {'num_finish': [], 'wall_crash': [], 'conflicts': [], 'iters': [], 'rewards': []}]
    for i in range(len(Algorithm)):
        for exp in range(experiments_times):
            env = Inventory(NumOfPlayers, Algorithm[i], IfSnippet=False, IfGoal=True, taskIndice=task_indice)
            frames = []
            frames.append(env.layout_)
            print ('This is the experiment {}!'.format(exp))
            print(frames[-1])
            for step in range(Steps):
                print ('Step {}'.format(step))
                last_mean_rewards = 0
                env.reset_layout()
                for iters in range(Iters):
                    rewards = env.__procudure__()
                    mean_rewards = last_mean_rewards + 1/(iters+1) * (np.sum(rewards) - last_mean_rewards)
                    if np.abs(mean_rewards - last_mean_rewards) < .1:
                        experiments[i]['iters'].append(iters+1)
                        experiments[i]['rewards'].append(np.sum(rewards))
                        print ('This is the iters num: {}'.format(iters+1))
                        break
                    elif iters == Iters - 1:
                        experiments[i]['iters'].append(iters+1)
                        experiments[i]['rewards'].append(np.sum(rewards))
                        print ('This is the iters num: {}'.format(iters+1))
                    else:
                        env.conflicts = 0
                        env.wall_crash = 0
                    last_mean_rewards = mean_rewards
                conflicts += env.conflicts
                wall_crash += env.wall_crash
                env.update_layout()
                frames.append(env.layout_)
                env.update_start_states(env.states)
                num_finish = sum(env.check_assignments())
                print ('The number of successful assignments is {}.'.format(num_finish))
                print ('The number of wall crash is {}.'.format(wall_crash))
                print ('The number of conflicts is {}.\n'.format(conflicts))
                if num_finish == 60:
                    experiments[i]['num_finish'].append(step)
                    experiments[i]['wall_crash'].append(wall_crash)
                    experiments[i]['conflicts'].append(conflicts)
                    break
    np.save('./mffp_exp3_iter1.npy', np.array(experiments[0]['iters']))
    np.save('./jsfp_exp3_iter1.npy', np.array(experiments[1]['iters']))
    np.save('./mffp_exp3_seed1.npy', np.array([experiments[0]['num_finish'], experiments[0]['wall_crash'], experiments[0]['conflicts'], experiments[0]['rewards']]))
    np.save('./jsfp_exp3_seed1.npy', np.array([experiments[1]['num_finish'], experiments[1]['wall_crash'], experiments[1]['conflicts'], experiments[1]['rewards']]))
else:
    print ('Please input the correct mode from 0, 1 and 2!')
