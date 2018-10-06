import numpy as np


class Agent:

    def __init__(self, NumActions, NumAgents, type, name):
        self.name = name
        self.type = type
        self.t = 0
        self.task = 0
        self.NumActions = NumActions
        self.NumAgents = NumAgents
        if self.type == 'actor_critic':
            self.Qj = np.random.rand(NumActions)
            self.policy = np.random.rand(NumActions)
            self.policy = self.policy / np.sum(self.policy)
            # self.action = np.random.choice(NumActions, p=self.policy)
            self.action = 0
            self.C_alpha = 1.
            self.C_lambda = 1.
            self.rho_pi = .498
            self.rho_lambda = .501
            self.rho_alpha = 1.
            self.c = np.zeros((NumActions, ), dtype=np.float32)
        elif self.type == 'joint_actions':
            self.Qj = np.random.rand(NumActions)
            # self.action = np.random.choice(NumActions)
            self.action = 0
            self.alpha = .7
            self.rho = .8
        elif self.type == 'mean_field':
            self.Qj = np.random.rand(NumActions, NumActions)
            # self.action = np.random.choice(NumActions)
            self.action = 0
            self.enermy_action_hist = np.zeros((NumActions, ), dtype=np.float32)
            self.c = .1
            self.d = .0005
        elif self.type == 'sample_average':
            self.Qj = np.random.rand(NumActions)
            # self.action = np.random.choice(NumActions)
            self.action = 0
            self.epsilon = 0.1
            self.c = np.zeros((NumActions, ), dtype=np.float32)

    def erase_memory(self):
        self.t = 0
        if self.type == 'actor_critic':
            self.Qj = np.random.rand(self.NumActions)
            self.policy = np.random.rand(self.NumActions)
            self.policy = self.policy / np.sum(self.policy)
            # self.action = np.random.choice(self.NumActions, p=self.policy)
            self.action = 0
            self.c = np.zeros((self.NumActions, ), dtype=np.int32)
        elif self.type == 'joint_actions':
            self.Qj = np.random.rand(self.NumActions)
            # self.action = np.random.choice(self.NumActions)
            self.action = 0
        elif self.type == 'mean_field':
            self.Qj = np.random.rand(self.NumActions, self.NumActions)
            # self.action = np.random.choice(self.NumActions)
            self.action = 0
            self.enermy_action_hist = np.zeros((self.NumActions, ), dtype=np.float32)
        elif self.type == 'sample_average':
            self.Qj = np.random.rand(self.NumActions)
            # self.action = np.random.choice(self.NumActions)
            self.action = 0

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
        elif self.type == 'mean_field':
            self.rho = 1 / (self.t + 1)**self.d
            self.alpha = 1 / (self.t + 1)**self.c
            enermy_actions = []
            for agent in agents:
                if agent.name != self.name:
                    enermy_actions.append(agent.action)
            enermy_action_curr = int(round(np.mean(enermy_actions)))
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
        elif self.type == 'sample_average':
            self.c[self.action] += 1
            self.alpha = 1 / self.c[self.action]
            self.Qj[self.action] = (1 - self.alpha) * self.Qj[self.action] + self.alpha * rewards[self.name]
            if np.random.rand() < self.epsilon:
                self.action = np.random.choice(self.NumActions)
            else:
                self.action = np.argmax(self.Qj)
        self.t += 1
