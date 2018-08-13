import numpy as np
import time
from agent import *
from collections import Counter
import matplotlib.pyplot as plt

np.random.seed(2)

class SimpleCongestion:

    def __init__(self, NumRoad, NumAgents, agentType, roads):
        self.agentType = agentType
        self.roads = roads
        self.agents = [Agent(NumRoad, NumAgents, self.agentType, i) for i in range(NumAgents)]

    def costFunc(self):
        costs = []
        for i in range(len(self.roads)):
            costs.append(self.roads[i][0] * self.carsOnRoads[i]**2 + self.roads[i][1] * self.carsOnRoads[i] + self.roads[i][2])
        return costs

    def __getCost__(self):
        self.carsOnRoads = Counter()
        self.roadsChoice = [agent.action for agent in self.agents]
        for choice in self.roadsChoice:
            self.carsOnRoads[choice] += 1
        return self.costFunc()

    def __getReward__(self):
        self.costs = self.__getCost__()
        self.rewards = []
        for choice in self.roadsChoice:
            self.rewards.append(- self.costs[choice])

    def __procudure__(self):
        self.__getReward__()
        agents = self.agents.copy()
        for agent in self.agents:
            agent.__update__(agents, self.rewards)
        return self.costs, [self.carsOnRoads[road] for road in range(len(self.roads))]


class StrategyGame:

    def __init__(self, agentType):
        self.agentType = agentType
        self.payoffs = np.array([[(3, 3), (0, 4)], [(4, 0), (1, 1)]])
        self.agents = [Agent(2, 2, self.agentType, i) for i in range(2)]

    def __getReward__(self):
        self.rewards = self.payoffs[self.agents[0].action, self.agents[1].action]

    def __procudure__(self):
        self.__getReward__()
        self.policies = []
        agents = self.agents.copy()
        for agent in self.agents:
            agent.__update__(agents, self.rewards)
            self.policies.append(agent.action)
        return self.policies


'''
conflicts: - 3
wall: - 7
good pos: 0
keep stay: - 1
goal: + 10
'''
class Inventory:

  def __init__(self, NumAgents, agentType, IfSnippet, IfGoal, taskIndice):
      self.start_states = []
      self.IfGoal = IfGoal
      self.reset_layout()
      if IfSnippet:
          for _ in range(NumAgents):
              sample = np.array([0, 0])
              while self.layout[sample[0], sample[1]] < 0 or self.if_duplicate(sample):
                  sample = tuple(np.random.randint(low=1, high=self.layout.shape[1]-1, size=(2,)).tolist())
              self.start_states.append(sample)
      else:
          for i in range(13):
              self.start_states.append(np.array([i+2, 1]))
              self.start_states.append(np.array([i+2, 19]))
          for i in range(17):
              self.start_states.append(np.array([1, i+2]))
              self.start_states.append(np.array([15, i+2]))
      self.reset_states()
      self.update_layout()
      self._number_of_states = np.prod(self.layout.shape)
      self.agentType = agentType
      self.NumAgents = NumAgents
      self.actionNum = 5
      self.agents = [Agent(self.actionNum, self.NumAgents, self.agentType, i) for i in range(NumAgents)]
      racks = [(5, 5), (5, 10), (5, 15), (11, 5), (11, 10), (11, 15)]
      self.tasks = []
      for i in range(NumAgents):
          self.tasks.append(racks[taskIndice[i]])
      self.assign_tasks()

  def assign_tasks(self):
      cnt = Counter()
      for i in range(len(self.agents)):
          self.agents[i].task = self.tasks[i]
          cnt[self.tasks[i]] += 1
      print (cnt.most_common(6))

  def if_duplicate(self, curr_pos):
      flag = False
      for pos in self.start_states:
          if np.all(curr_pos == pos):
              flag = True
      return flag

  def number_of_states(self):
      return self._number_of_states

  def update_layout(self):
      for state in self.states:
          self.layout[state[0], state[1]] += 1
          self.layout_[state[0], state[1]] += 1

  def plot_grid(self):
      plt.figure()
      plt.imshow(self.layout, vmin=-1, vmax=3)
      plt.xticks([])
      plt.yticks([])
      plt.colorbar()

  def update_start_states(self, states):
      self.start_states = states
      for agent in self.agents: agent.erase_memory()
      # self.assign_tasks()

  def reset_states(self):
      self.states = self.start_states

  def reset_layout(self):
      # -1: wall
      # 0: empty, episode continues
      if not self.IfGoal:
          self.layout = np.array([
            [-1, -1,  -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,  -1, -1, -1, -1],
            [-1,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0, -1],
            [-1,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0, -1],
            [-1,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0, -1],
            [-1,  0,   0, -1, -1,  0,  0, -1, -1, -1,  0,   0,  -1,  -1,  0,  0, -1],
            [-1,  0,   0, -1, -1,  0,  0, -1, -1, -1,  0,   0,  -1,  -1,  0,  0, -1],
            [-1,  0,   0, -1, -1,  0,  0, -1, -1, -1,  0,   0,  -1,  -1,  0,  0, -1],
            [-1,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0, -1],
            [-1,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0, -1],
            [-1,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0, -1],
            [-1,  0,   0, -1, -1,  0,  0, -1, -1, -1,  0,   0,  -1,  -1,  0,  0, -1],
            [-1,  0,   0, -1, -1,  0,  0, -1, -1, -1,  0,   0,  -1,  -1,  0,  0, -1],
            [-1,  0,   0, -1, -1,  0,  0, -1, -1, -1,  0,   0,  -1,  -1,  0,  0, -1],
            [-1,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0, -1],
            [-1,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0, -1],
            [-1,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0, -1],
            [-1, -1,  -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,  -1, -1, -1, -1]
          ])
          self.layout_ = np.array([
            [-1, -1,  -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,  -1, -1, -1, -1],
            [-1,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0, -1],
            [-1,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0, -1],
            [-1,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0, -1],
            [-1,  0,   0, -1, -1,  0,  0, -1, -1, -1,  0,   0,  -1,  -1,  0,  0, -1],
            [-1,  0,   0, -1, -1,  0,  0, -1, -1, -1,  0,   0,  -1,  -1,  0,  0, -1],
            [-1,  0,   0, -1, -1,  0,  0, -1, -1, -1,  0,   0,  -1,  -1,  0,  0, -1],
            [-1,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0, -1],
            [-1,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0, -1],
            [-1,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0, -1],
            [-1,  0,   0, -1, -1,  0,  0, -1, -1, -1,  0,   0,  -1,  -1,  0,  0, -1],
            [-1,  0,   0, -1, -1,  0,  0, -1, -1, -1,  0,   0,  -1,  -1,  0,  0, -1],
            [-1,  0,   0, -1, -1,  0,  0, -1, -1, -1,  0,   0,  -1,  -1,  0,  0, -1],
            [-1,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0, -1],
            [-1,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0, -1],
            [-1,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0, -1],
            [-1, -1,  -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,  -1, -1, -1, -1]
          ])
      else:
           self.layout = np.array([
             [-1, -1,  -1,  -1,  -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,  -1, -1, -1, -1, -1, -1],
             [-1,  0,   0,   0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0,  0,  0, -1],
             [-1,  0,   0,   0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0,  0,  0, -1],
             [-1,  0,   0,   0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0,  0,  0, -1],
             [-1,  0,   0,  -1,   0,  0,  0, -1, -1,  0,  0,  0, -1,  -1,   0,   0,  0, -1,  0,  0, -1],
             [-1,  0,   0,   0,   0, 70,  0,  0,  0,  0, 70,  0,  0,   0,   0,  70,  0,  0,  0,  0, -1],
             [-1,  0,   0,  -1,   0,  0,  0, -1, -1,  0,  0,  0, -1,  -1,   0,   0,  0, -1,  0,  0, -1],
             [-1,  0,   0,   0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0,  0,  0, -1],
             [-1,  0,   0,  -1,  -1,  0, -1, -1, -1, -1,  0, -1, -1,  -1,  -1,   0, -1, -1,  0,  0, -1],
             [-1,  0,   0,   0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0,  0,  0, -1],
             [-1,  0,   0,  -1,   0,  0,  0, -1, -1,  0,  0,  0, -1,  -1,   0,   0, -1, -1,  0,  0, -1],
             [-1,  0,   0,   0,   0, 70,  0,  0,  0,  0, 70,  0,  0,   0,   0,  70,  0,  0,  0,  0, -1],
             [-1,  0,   0,  -1,   0,  0,  0, -1, -1,  0,  0,  0, -1,  -1,   0,   0,  0, -1,  0,  0, -1],
             [-1,  0,   0,   0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0,  0,  0, -1],
             [-1,  0,   0,   0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0,  0,  0, -1],
             [-1,  0,   0,   0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0,  0,  0, -1],
             [-1, -1,  -1,  -1,  -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,  -1, -1, -1, -1, -1, -1]
           ])
           self.layout_ = np.array([
             [-1, -1,  -1,  -1,  -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,  -1, -1, -1, -1, -1, -1],
             [-1,  0,   0,   0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0,  0,  0, -1],
             [-1,  0,   0,   0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0,  0,  0, -1],
             [-1,  0,   0,   0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0,  0,  0, -1],
             [-1,  0,   0,  -1,   0,  0,  0, -1, -1,  0,  0,  0, -1,  -1,   0,   0,  0, -1,  0,  0, -1],
             [-1,  0,   0,   0,   0,  5,  0,  0,  0,  0,  5,  0,  0,   0,   0,   5,  0,  0,  0,  0, -1],
             [-1,  0,   0,  -1,   0,  0,  0, -1, -1,  0,  0,  0, -1,  -1,   0,   0,  0, -1,  0,  0, -1],
             [-1,  0,   0,   0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0,  0,  0, -1],
             [-1,  0,   0,  -1,  -1,  0, -1, -1, -1, -1,  0, -1, -1,  -1,  -1,   0, -1, -1,  0,  0, -1],
             [-1,  0,   0,   0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0,  0,  0, -1],
             [-1,  0,   0,  -1,   0,  0,  0, -1, -1,  0,  0,  0, -1,  -1,   0,   0,  0, -1,  0,  0, -1],
             [-1,  0,   0,   0,   0,  5,  0,  0,  0,  0,  5,  0,  0,   0,   0,   5,  0,  0,  0,  0, -1],
             [-1,  0,   0,  -1,   0,  0,  0, -1, -1,  0,  0,  0, -1,  -1,   0,   0,  0, -1,  0,  0, -1],
             [-1,  0,   0,   0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0,  0,  0, -1],
             [-1,  0,   0,   0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0,  0,  0, -1],
             [-1,  0,   0,   0,   0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,  0,  0,  0,  0, -1],
             [-1, -1,  -1,  -1,  -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,  -1, -1, -1, -1, -1, -1]
           ])

  def distance(self, a, b):
      return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

  def check_assignments(self):
      return [np.all(self.states[i] == self.agents[i].task) for i in range(len(self.states))]

  def __getReward__(self):
      self.rewards = []
      self.states_ = []
      def getStates(action, state):
          y, x = state
          if action == 0:    # up
              new_state = (y - 1, x)
          elif action == 1:  # right
              new_state = (y, x + 1)
          elif action == 2:  # down
              new_state = (y + 1, x)
          elif action == 3:  # left
              new_state = (y, x - 1)
          elif action == 4:  # keep stay
              new_state = (y, x)
          else:
              raise ValueError("Invalid action: {} is not 0, 1, 2, 3 or 4.".format(action))
          if self.layout[new_state[0], new_state[1]] == -1:
              new_state = (y, x, '')
          self.states_.append(new_state)
      def getReward(state, new_state, agent):
          if self.layout[new_state[0], new_state[1]] == 1:
              if len(new_state) == 2:
                  if np.all(new_state == state): # keep stay
                      reward = - 1.
                  else: # move to a good pos
                      reward = 0.
              elif len(new_state) > 2: # wall
                  reward = - 7.
          elif self.layout[new_state[0], new_state[1]] == 71:
              if np.all(agent.task == new_state):
                  reward = 10.
              else: # penalty - 5
                  if len(new_state) == 2:
                      if np.all(new_state == state): # keep stay
                          reward = - 6.
                      else: # move to a good pos
                          reward = - 5.
                  elif len(new_state) > 2: # wall
                      reward = - 12.
          elif 60 >= self.layout[new_state[0], new_state[1]] > 1: # conflict with other agents
              if len(new_state) == 2:
                  if np.all(new_state == state): # keep stay
                      reward = - 4.
                  else: # move to a good pos
                      reward = - 3.
              elif len(new_state) > 2: # wall
                  reward = - 10.
          elif self.layout[new_state[0], new_state[1]] > 71:
              if np.all(agent.task == new_state):
                  reward = 7.
              else: # penalty - 5
                  if len(new_state) == 2:
                      if np.all(new_state == state): # keep stay
                          reward = - 9.
                      else: # move to a good pos
                          reward = - 8.
                  elif len(new_state) > 2: # wall
                      reward = - 15.
          reward -= 0.9 * self.distance(agent.task, new_state)
          return reward
      for i in range(len(self.agents)):
          getStates(self.agents[i].action, self.states[i])
      self._states = self.states
      states = []
      for i in range(len(self.states)):
          if len(self.states_[i]) > 2:
              states.append(self.states_[i][:2])
          else:
              states.append(self.states_[i])
      self.states = states
      del states
      self.reset_layout()
      self.update_layout()
      # all of new states have been drawn on the layout
      for i in range(len(self.agents)):
          reward = getReward(self._states[i], self.states_[i], self.agents[i])
          self.rewards.append(reward)

  def __procudure__(self, t):
      self.reset_layout()
      self.reset_states()
      self.update_layout()
      self.__getReward__()
      agents = self.agents.copy()
      for agent in self.agents:
          agent.__update__(agents, self.rewards)
      # if self.IfGoal:
      #     indice = []
      #     for i in range(len(self.rewards)):
      #         if self.rewards[i] >= 7:
      #             indice.append(i)
      #     agents = []
      #     states = []
      #     rewards = []
      #     states_ = []
      #     for j in range(len(self.agents)):
      #         if not (j in indice):
      #             agents.append(self.agents[j])
      #             states.append(self.states[j])
      #             rewards.append(self.rewards[j])
      #             states_.append(self.states_[j])
      #     self.agents = agents
      #     self.states = states
      #     self.rewards = rewards
      #     self.states_ = states_
      return self.rewards