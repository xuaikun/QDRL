from __future__ import print_function
import imp
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torch.nn import functional as F
from GA_code import GA

print(torch.__version__)
class MemoryDNN:
    def __init__(
        self,
        net,
        learning_rate = 0.01,
        training_interval=10,
        batch_size=100,
        memory_size=1000,
        output_graph=False
    ):

        self.net = net
        self.training_interval = training_interval   
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.enumerate_actions = []
        self.memory_counter = 1
        self.cost_his = []
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))    

        self._build_net()

    def _build_net(self): 
        self.model = nn.Sequential(
                nn.Linear(self.net[0], self.net[1]), # input layer
                nn.ReLU(), # 
                nn.Linear(self.net[1], self.net[2]), # hidden layer
                nn.ReLU(), #
                nn.Linear(self.net[2], self.net[3]), # output layer
                nn.Sigmoid() # 
        )
    def remember(self, h, m):
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))

        self.memory_counter += 1

    def encode(self, h, m):
        self.remember(h, m)
        if self.memory_counter % self.training_interval == 0:
            self.learn() #

    def learn(self): # training dnn
        if self.memory_counter > self.memory_size: # 
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else: # 
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        h_train = torch.Tensor(batch_memory[:, 0: self.net[0]])
        m_train = torch.Tensor(batch_memory[:, self.net[0]:])

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr,betas = (0.09,0.999),weight_decay=0.0001) 
        criterion = nn.BCEWithLogitsLoss() 
        self.model.train()
        optimizer.zero_grad()
        predict = self.model(h_train)
        loss = criterion(predict, m_train)
        loss.backward()
        optimizer.step()

        self.cost = loss.item()
        assert(self.cost > 0)
        self.cost_his.append(self.cost)

    def decode(self, h, k = 1, mode = 'OP'):
        h = torch.Tensor(h[np.newaxis, :])
        self.model.eval()
        m_pred = self.model(h)
        m_pred = m_pred.detach().numpy()

        if mode is 'OP':
            return self.knm(m_pred[0], k)
        elif mode is 'KNN':
            return self.knn(m_pred[0], k)
        elif mode == 'GA':
            return self.GAG(m_pred[0], k) # 
        elif mode == 'GA+OP':
            return self.GAOP(m_pred[0], k) # 
        else:
            print("The action selection must be 'OP' or 'KNN' or 'GA' or 'GA+OP'")

    def GAG(self, m, k = 1):
        # return k order-preserving binary actions
        m_list = []
        # generate the ﬁrst binary ofﬂoading decision with respect to equation (8)
        m_list.append(1*(m>0.5))
        M = m_list[0]
        N = len(m_list[0])
        M_temp = np.zeros(len(M))
        for j in range(len(M)): #
            M_temp[j] = 1 if M[j] == 0 else 0
        
        popDNA = np.zeros((k, N)) # 
        for i in range(k): # 
            if i%2 == 0:
                popDNA[i][0:] = M
            else:
                popDNA[i][0:] = M_temp
        m_list = GA(popDNA)
        m_list_temp = []
        for i in range(k):
            m_list_temp.append(m_list[i][0:])
        return m_list_temp

    def GAOP(self, m, k = 1):
        ################### GA+OP ########################
        N = len(m)
        # return k order-preserving binary actions
        popDNA = self.knm(m, k)
        popDNA_copy = popDNA.copy()
        new_type_popDNA = np.zeros((k, N)) #
        for i in range(k):
            new_type_popDNA[i][0:] = popDNA_copy[i]
        m_list = GA(new_type_popDNA)
        m_list_temp = []
        for i in range(k):
            m_list_temp.append(m_list[i][0:])
            m_list_temp.append(popDNA[i][0:])
        return m_list_temp
        

    def knm(self, m, k = 1):
        m_list = []
        m_list.append(1*(m>0.5))

        if k > 1:
            m_abs = abs(m-0.5)
            idx_list = np.argsort(m_abs)[:k-1]
            for i in range(k-1):
                if m[idx_list[i]] >0.5:
                    m_list.append(1*(m - m[idx_list[i]] > 0))
                else:
                    m_list.append(1*(m - m[idx_list[i]] >= 0))

        return m_list

    def knn(self, m, k = 1):
        if len(self.enumerate_actions) is 0:
            import itertools
            self.enumerate_actions = np.array(list(map(list, itertools.product([0, 1], repeat=self.net[3]))))

        sqd = ((self.enumerate_actions - m)**2).sum(1)
        idx = np.argsort(sqd)
        return self.enumerate_actions[idx[:k]]

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his))*self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.show()


class alpha_MemoryDNN:
    def __init__(
        self,
        net,
        learning_rate = 0.01,
        training_interval=10,
        batch_size=100,
        memory_size=1000,
        output_graph=False
    ):

        self.net = net
        self.training_interval = training_interval   
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.enumerate_actions = []

        self.memory_counter = 1
        self.cost_his = []

        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        self._build_net()

    def _build_net(self):
        self.model = nn.Sequential(
                nn.Linear(self.net[0], self.net[1]), 
                # nn.ReLU(), 
                nn.Tanh(),
                nn.Linear(self.net[1], self.net[2]), 
                # nn.ReLU(), #
                nn.Tanh(),
                nn.Linear(self.net[2], self.net[3]), #
                nn.Sigmoid() #
                # F.softmax()
                # nn.Softmax() #
                # nn.Tanh()
        )
    def remember(self, h, m):
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))

        self.memory_counter += 1

    def encode(self, h, m):
        self.remember(h, m)
        if self.memory_counter % self.training_interval == 0:
            self.learn() 

    def learn(self): 
        if self.memory_counter > self.memory_size: 
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else: 
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        h_train = torch.Tensor(batch_memory[:, 0: self.net[0]])
        m_train = torch.Tensor(batch_memory[:, self.net[0]:])

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr,betas = (0.09,0.999),weight_decay=0.0001) 
        criterion = nn.BCELoss()
        self.model.train()
        optimizer.zero_grad()
        predict = self.model(h_train)
        loss = criterion(predict, m_train)
        loss.backward()
        optimizer.step()

        self.cost = loss.item()
        assert(self.cost > 0)
        self.cost_his.append(self.cost)

    def decode(self, h):
        h = torch.Tensor(h[np.newaxis, :])

        self.model.eval()
        m_pred = self.model(h)
        m_pred = m_pred.detach().numpy()

        return m_pred[0]

    def plot_cost(self): 
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his))*self.training_interval, self.cost_his)
        plt.ylabel('alpha_Training Loss')
        plt.xlabel('Time Frames')
        plt.show()

class tau_MemoryDNN:
    def __init__(
        self,
        net,
        learning_rate = 0.01,
        training_interval=10,
        batch_size=100,
        memory_size=1000,
        output_graph=False
    ):

        self.net = net
        self.training_interval = training_interval 
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.enumerate_actions = []

        self.memory_counter = 1

        self.cost_his = []
        
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        self._build_net()

    def _build_net(self): 
        self.model = nn.Sequential(
                nn.Linear(self.net[0], self.net[1]), #
                nn.ReLU(), # 
                #nn.Tanh(),
                nn.Linear(self.net[1], self.net[2]), # 
                nn.ReLU(), # 
                # nn.Tanh(),
                nn.Linear(self.net[2], self.net[3]), # 
                # nn.Sigmoid() # 
                # F.softmax()
                nn.Softmax(dim=1) #
                # nn.Tanh()
        )
    def remember(self, h, m):
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))

        self.memory_counter += 1

    def encode(self, h, m):
        self.remember(h, m)
        if self.memory_counter % self.training_interval == 0:
            self.learn() # 

    def learn(self): # training dnn
        if self.memory_counter > self.memory_size: 
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else: 
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        h_train = torch.Tensor(batch_memory[:, 0: self.net[0]])
        m_train = torch.Tensor(batch_memory[:, self.net[0]:])

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr,betas = (0.09,0.999),weight_decay=0.0001) 
        criterion = nn.BCEWithLogitsLoss() 
        self.model.train()
        optimizer.zero_grad()
        predict = self.model(h_train)
        loss = criterion(predict, m_train)
        loss.backward()
        optimizer.step()

        self.cost = loss.item()
        assert(self.cost > 0)
        self.cost_his.append(self.cost)

    def decode(self, h):
        h = torch.Tensor(h[np.newaxis, :])

        self.model.eval()
        m_pred = self.model(h)
        m_pred = m_pred.detach().numpy()

        return m_pred[0]

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his))*self.training_interval, self.cost_his)
        plt.ylabel('tau_Training Loss')
        plt.xlabel('Time Frames')
        plt.show()
