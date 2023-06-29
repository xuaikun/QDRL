#  ################################################################
#  该文件包含DROO的主要操作，包括构建DNN网络，保持数据样本，训练DNN，
#  生成量化的二进制卸载决策
#  ###################################################################

from __future__ import print_function
import imp
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torch.nn import functional as F
from GA_code import GA

print(torch.__version__)

# 以下公式来自论文：Deep Reinforcement Learning for Online Offloading in Wireless Powered Mobile-Edge Computing Networks
# 用于存储DNN网络
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
        self.training_interval = training_interval      # 间隔多久训练一次DNN，即训练间隔
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size

        # 保存所有二元决策（卸载决策），用作训练DNN的数据集
        self.enumerate_actions = []

        #######  不同的网络，对应参数需要不同 #####

        # 记录内存条目（类似于指向哪块存储空间）
        self.memory_counter = 1
        # 保存训练成本
        self.cost_his = []

        # 初始化内存为0
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))    

        # 构建DNN网络
        self._build_net()

    def _build_net(self): # 构建DNN网络
        self.model = nn.Sequential(
                nn.Linear(self.net[0], self.net[1]), # 输入层
                nn.ReLU(), # 激活函数
                nn.Linear(self.net[1], self.net[2]), # 隐藏层
                nn.ReLU(), # 激活函数
                nn.Linear(self.net[2], self.net[3]), # 输出层
                nn.Sigmoid() # 激活函数
        )
    def remember(self, h, m):
        # 利用新的数据替换陈旧的数据
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))

        self.memory_counter += 1

    def encode(self, h, m):
        # 对条目进行编码
        self.remember(h, m)
        # 每self.training_interval步训练一次DNN
        if self.memory_counter % self.training_interval == 0:
            self.learn() # 训练DNN

    def learn(self): # 训练DNN
        # 从存储的所有数据中采样部分数据用于训练网络（采样批处理）
        if self.memory_counter > self.memory_size: # 分配的内存未存满数据
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else: # 分配的内存已存满数据
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        # 对应训练数据和标签
        h_train = torch.Tensor(batch_memory[:, 0: self.net[0]])
        m_train = torch.Tensor(batch_memory[:, self.net[0]:])


        # 训练DNN
        # optimizer = optim.Adam(self.model.parameters(), lr=self.lr,betas = (0.09,0.999),weight_decay=0.0001) 
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr,betas = (0.09,0.999),weight_decay=0.0001) 
        # criterion = nn.BCELoss()
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
        # 输入的张量时具有批处理的大小
        h = torch.Tensor(h[np.newaxis, :])

        self.model.eval()
        m_pred = self.model(h)
        m_pred = m_pred.detach().numpy()

        # 3种量化卸载决策变量的方法
        if mode is 'OP':
            return self.knm(m_pred[0], k)
        elif mode is 'KNN':
            return self.knn(m_pred[0], k)
        elif mode == 'GA':
            return self.GAG(m_pred[0], k) # 'GA' 纯GA方案，结合方案：'GA+OP',
        elif mode == 'GA+OP':
            return self.GAOP(m_pred[0], k) # 'GA' 纯GA方案
        else:
            print("The action selection must be 'OP' or 'KNN' or 'GA' or 'GA+OP'")

    def GAG(self, m, k = 1):
        ################### 纯GA方案 ########################
        # return k order-preserving binary actions
        m_list = []
        # generate the ﬁrst binary ofﬂoading decision with respect to equation (8)
        m_list.append(1*(m>0.5))
        M = m_list[0]
        N = len(m_list[0])
        M_temp = np.zeros(len(M))
        for j in range(len(M)): # 与M相反的卸载方式
            M_temp[j] = 1 if M[j] == 0 else 0
        
        popDNA = np.zeros((k, N)) # 定义种群
        for i in range(k): # 生成种群
            if i%2 == 0:
                popDNA[i][0:] = M
            else:
                popDNA[i][0:] = M_temp
        # print("m_list =", m_list, type(m_list), m_list[0], type(m_list[0]))
        m_list = GA(popDNA)
        # print("m_list =", m_list, type(m_list))
        m_list_temp = []
        for i in range(k):
            m_list_temp.append(m_list[i][0:])
        # print("m_list_temp =", m_list_temp, type(m_list_temp))
        # np,show()
        return m_list_temp

    def GAOP(self, m, k = 1):
        ################### GA+OP ########################
        N = len(m)
        # return k order-preserving binary actions
        popDNA = self.knm(m, k)
        popDNA_copy = popDNA.copy()
        new_type_popDNA = np.zeros((k, N)) # 定义种群
        for i in range(k):
            new_type_popDNA[i][0:] = popDNA_copy[i]
        # print("popDNA =", popDNA, type(popDNA))
        # print("new_type_popDNA =", new_type_popDNA, type(new_type_popDNA))
        m_list = GA(new_type_popDNA)
        # print("m_list =", m_list, type(m_list))
        m_list_temp = []
        for i in range(k):
            m_list_temp.append(m_list[i][0:])
            m_list_temp.append(popDNA[i][0:])
        # print("m_list_temp =", m_list_temp, type(m_list_temp))
        # np,show()
        return m_list_temp
        

    def knm(self, m, k = 1):
        # 返回k个保序的卸载决策
        m_list = []
        # 根据公式 (8)生成第一个卸载决策,Deep Reinforcement Learning for Online Offloading in Wireless Powered Mobile-Edge Computing Networks
        m_list.append(1*(m>0.5))

        if k > 1:
            # 根据公式 (9)生成剩余的K-1个卸载决策，Deep Reinforcement Learning for Online Offloading in Wireless Powered Mobile-Edge Computing Networks
            m_abs = abs(m-0.5)
            idx_list = np.argsort(m_abs)[:k-1]
            for i in range(k-1):
                if m[idx_list[i]] >0.5:
                    # 设置 \hat{x}_{t,(k-1)} 为 0
                    m_list.append(1*(m - m[idx_list[i]] > 0))
                else:
                    # 设置 \hat{x}_{t,(k-1)} 为 1
                    m_list.append(1*(m - m[idx_list[i]] >= 0))

        return m_list

    def knn(self, m, k = 1):
        # 列出2的n次方个卸载决策
        if len(self.enumerate_actions) is 0:
            import itertools
            self.enumerate_actions = np.array(list(map(list, itertools.product([0, 1], repeat=self.net[3]))))

        # 2-范数
        sqd = ((self.enumerate_actions - m)**2).sum(1)
        idx = np.argsort(sqd)
        return self.enumerate_actions[idx[:k]]

    def plot_cost(self): # 训练误差图
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
        self.training_interval = training_interval      # 间隔多久训练一次DNN，即训练间隔
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size

        # 保存所有二元决策（卸载决策），用作训练DNN的数据集
        self.enumerate_actions = []

        #######  不同的网络，对应参数需要不同 #####

        # 记录内存条目（类似于指向哪块存储空间）
        self.memory_counter = 1
        # 保存训练成本
        self.cost_his = []

        # 初始化内存为0
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))
  
        # 构建DNN网络
        self._build_net()

    def _build_net(self): # 构建DNN网络
        self.model = nn.Sequential(
                nn.Linear(self.net[0], self.net[1]), # 输入层
                # nn.ReLU(), # 激活函数
                nn.Tanh(),
                nn.Linear(self.net[1], self.net[2]), # 隐藏层
                # nn.ReLU(), # 激活函数
                nn.Tanh(),
                nn.Linear(self.net[2], self.net[3]), # 输出层
                nn.Sigmoid() # 激活函数
                # F.softmax()
                # nn.Softmax() # 求概率，分类才有用
                # nn.Tanh()
        )
    def remember(self, h, m):
        # 利用新的数据替换陈旧的数据
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))

        self.memory_counter += 1

    def encode(self, h, m):
        # 对条目进行编码
        self.remember(h, m)
        # 每self.training_interval步训练一次DNN
        if self.memory_counter % self.training_interval == 0:
            self.learn() # 训练DNN

    def learn(self): # 训练DNN
        # 从存储的所有数据中采样部分数据用于训练网络（采样批处理）
        if self.memory_counter > self.memory_size: # 分配的内存未存满数据
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else: # 分配的内存已存满数据
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        # 对应训练数据和标签
        h_train = torch.Tensor(batch_memory[:, 0: self.net[0]])
        m_train = torch.Tensor(batch_memory[:, self.net[0]:])


        # 训练DNN
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
        # 输入的张量时具有批处理的大小
        h = torch.Tensor(h[np.newaxis, :])

        self.model.eval()
        m_pred = self.model(h)
        m_pred = m_pred.detach().numpy()

        return m_pred[0]

    def plot_cost(self): # 训练误差图
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
        self.training_interval = training_interval      # 间隔多久训练一次DNN，即训练间隔
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size

        # 保存所有二元决策（卸载决策），用作训练DNN的数据集
        self.enumerate_actions = []

        #######  不同的网络，对应参数需要不同 #####

        # 记录内存条目（类似于指向哪块存储空间）
        self.memory_counter = 1

        # 保存训练成本
        self.cost_his = []
        
        # 初始化内存为0
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        # 构建DNN网络
        self._build_net()

    def _build_net(self): # 构建DNN网络
        self.model = nn.Sequential(
                nn.Linear(self.net[0], self.net[1]), # 输入层
                nn.ReLU(), # 激活函数
                #nn.Tanh(),
                nn.Linear(self.net[1], self.net[2]), # 隐藏层
                nn.ReLU(), # 激活函数
                # nn.Tanh(),
                nn.Linear(self.net[2], self.net[3]), # 输出层
                # nn.Sigmoid() # 激活函数
                # F.softmax()
                nn.Softmax(dim=1) # 求概率，分类才有用
                # nn.Tanh()
        )
    def remember(self, h, m):
        # 利用新的数据替换陈旧的数据
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))

        self.memory_counter += 1

    def encode(self, h, m):
        # 对条目进行编码
        self.remember(h, m)
        # 每self.training_interval步训练一次DNN
        if self.memory_counter % self.training_interval == 0:
            self.learn() # 训练DNN

    def learn(self): # 训练DNN
        # 从存储的所有数据中采样部分数据用于训练网络（采样批处理）
        if self.memory_counter > self.memory_size: # 分配的内存未存满数据
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else: # 分配的内存已存满数据
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        # 对应训练数据和标签
        h_train = torch.Tensor(batch_memory[:, 0: self.net[0]])
        m_train = torch.Tensor(batch_memory[:, self.net[0]:])


        # 训练DNN
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr,betas = (0.09,0.999),weight_decay=0.0001) 
        # criterion = nn.BCELoss()
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
        # 输入的张量时具有批处理的大小
        h = torch.Tensor(h[np.newaxis, :])

        self.model.eval()
        m_pred = self.model(h)
        m_pred = m_pred.detach().numpy()

        return m_pred[0]

    def plot_cost(self): # 训练误差图
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his))*self.training_interval, self.cost_his)
        plt.ylabel('tau_Training Loss')
        plt.xlabel('Time Frames')
        plt.show()