import numpy as np


class Agent:

    def __init__(self, NumActions, type, name):
        self.name = name
        self.type = type
        self.t = 0
        self.task = 0
        self.NumActions = NumActions
        if self.type == 'actor_critic':
            self.Qj = np.random.rand(NumActions)
            self.policy = np.random.rand(NumActions)
            self.policy = self.policy / np.sum(self.policy)
            self.action = np.random.choice(NumActions, p=self.policy)
            self.C_alpha = 1
            self.C_lambda = 1
            self.rho_pi = 0.48
            self.rho_lambda = 0.51
            self.rho_alpha = 1.
            self.c = np.zeros((NumActions, ))
        elif self.type == 'joint_actions':
            self.Qj = np.random.rand(NumActions)
            self.action = np.random.choice(NumActions)
            self.alpha = 0.2
            self.rho = 0.7
        elif self.type == 'mean_field':
            self.Qj = np.random.rand(NumActions, NumActions)
            self.action = np.random.choice(NumActions)
            # self.action = 0
            self.enermy_action_hist = np.zeros((NumActions, ), dtype=np.float32)
            # self.best_response_hist = np.zeros((NumActions, ), dtype=np.float32)
            self.c = 0.1
            self.d = 0.03

    def erase_memory(self):
        self.t = 0
        if self.type == 'actor_critic':
            self.Qj = np.random.rand(self.NumActions)
            self.policy = np.random.rand(self.NumActions)
            self.policy = self.policy / np.sum(self.policy)
            self.action = np.random.choice(self.NumActions, p=self.policy)
            self.c = np.zeros((self.NumActions, ))
        elif self.type == 'joint_actions':
            self.Qj = np.random.rand(self.NumActions)
            self.action = np.random.choice(self.NumActions)
        elif self.type == 'mean_field':
            self.Qj = np.random.rand(self.NumActions, self.NumActions)
            self.action = np.random.choice(self.NumActions)
            # self.action = 0
            self.enermy_action_hist = np.zeros((self.NumActions, ), dtype=np.float32)

    def __update__(self, agents, rewards):
        if self.type == 'actor_critic':
            self.c[self.action] += 1
            alpha = (self.C_alpha + self.t + 1)**(- self.rho_alpha)
            lambda_ = (self.C_lambda + self.c[self.action])**(- self.rho_lambda)
            tau = (np.max(self.Qj) - np.min(self.Qj)) / (self.rho_pi * np.log(self.t + 1) + 1e-10)
            beta = np.exp(self.Qj / tau - np.max(self.Qj / tau)) / np.sum(np.exp(self.Qj / tau - np.max(self.Qj / tau)))
            self.policy = (1 - alpha) * self.policy + alpha * beta
            self.Qj[self.action] = self.Qj[self.action] + lambda_ * (rewards[self.name] - self.Qj[self.action])
            self.action = np.random.choice(self.policy.shape[0], p=self.policy)
        elif self.type == 'joint_actions':
            bestResponse = np.where(self.Qj == np.max(self.Qj))[0].tolist()
            self.Qj[self.action] = (1 - self.rho) * self.Qj[self.action] + self.rho * rewards[self.name]
            if not (self.action in bestResponse):
                sudoDistr = np.random.rand(len(bestResponse))
                sudoDistr /= np.sum(sudoDistr)
                DistrVec = np.zeros(self.Qj.shape[0], dtype=np.float32)
                for i in range(self.Qj.shape[0]):
                    if i in bestResponse:
                        DistrVec[i] = sudoDistr[bestResponse.index(i)]
                DistrVec *= self.alpha
                DistrVec[self.action] += (1 - self.alpha)
                self.action = np.random.choice(self.Qj.shape[0], p=DistrVec)
            else:
                # print ('The last action is belonging to the current Best Response!')
                pass
        elif self.type == 'mean_field':
            self.rho = 1 / (self.t + 1)**self.d
            self.alpha = 1 / (self.t + 1)**self.c
            self.m = len(agents) - 1
            enermy_actions = []
            for agent in agents:
                if agent.name != self.name:
                    enermy_actions.append(agent.action)
            partial_enermy_actions = np.random.choice(enermy_actions, self.m, replace=False)
            enermy_action_curr = int(round(np.mean(partial_enermy_actions)))
            self.enermy_action_hist[enermy_action_curr] += 1
            enermy_actions_prob = np.exp(self.enermy_action_hist - np.max(self.enermy_action_hist)) / np.sum(np.exp(self.enermy_action_hist - np.max(self.enermy_action_hist)))
            q = self.Qj.dot(enermy_actions_prob)
            brc = np.where(q == np.max(q))[0].tolist()
            self.Qj[self.action, enermy_action_curr] = (1 - self.rho) * self.Qj[self.action, enermy_action_curr] + self.rho * rewards[self.name]
            if not (self.action in brc):
                sudoDistr = np.random.rand(len(brc))
                sudoDistr /= np.sum(sudoDistr)
                DistrVec = np.zeros(self.Qj.shape[0], dtype=np.float32)
                for i in range(self.Qj.shape[0]):
                    if i in brc:
                        DistrVec[i] = sudoDistr[brc.index(i)]
                DistrVec *= self.alpha
                DistrVec[self.action] += (1 - self.alpha)
                self.action = np.random.choice(self.NumActions, p=DistrVec)
            else:
                # print ('The last action is belonging to the current Best Response!')
                pass
        self.t += 1