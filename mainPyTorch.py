
import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy
import math

# 基于PyTorch实现 
from memoryPyTorch import MemoryDNN, alpha_MemoryDNN, tau_MemoryDNN # 做卸载决策所使用的神经网络模型（DNN），包含推理、训练、存储（输入，标签）对和采样数据
from optimization import bisection, cd_method, Mybisection, Mycd_method, alpha_Mybisection, alpha_Mycd_method # 资源分配

import time


def plot_rate(rate_his, rolling_intv=50, ylabel = 'Normalized Computation Rate'): # 生成图片
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    rate_array = np.asarray(rate_his)
    df = pd.DataFrame(rate_his)


    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15, 8))
#    rolling_intv = 20

    plt.plot(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values), 'b')
    plt.fill_between(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values), np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values), color = 'b', alpha = 0.2)
    plt.ylabel(ylabel)
    plt.xlabel('Time Frames')
    plt.show()

def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)

# generate racian fading channel with power h and Line of sight ratio factor
# replace it with your own channel generations when necessary
def racian_mec(h, factor):
    n = len(h)
    beta = np.sqrt(h*factor) # LOS channel amplitude
    sigma = np.sqrt(h*(1-factor)/2) # scattering sdv
    x = np.multiply(sigma*np.ones((n)),np.random.randn(n)) + beta*np.ones((n))
    y = np.multiply(sigma*np.ones((n)),np.random.randn(n))
    g = np.power(x,2) +  np.power(y,2)
    return g

if __name__ == "__main__":
    '''
        input_h:可以看作来自环境的状态集合
        r_list:表示来自环境奖励集合
        m_list:表示给环境的动作集合

        该算法从DNN生成K个卸载决策,并选择具有最大奖励的卸载决策。 
        奖励最大的卸载决策存储在内存中，进一步用于训练 DNN。 
        我们实现了自适应K.K = max(K, K_his[-memory_size])
    '''
    new_flag = False # 是否需要新产生数据，False是不需要产生新数据
    CD_gain_flag = False  # 是否更新CD_gain的值，False是不需要
    N = 10                       # 用户数量
    n = 500                    # 时间帧的数量（决定了整个系统运行多久）
    K = N                        # 初始化 K = N
    decoder_mode = 'GA+OP'          # 量化卸载决策的方式可选 'OP' (Order-preserving) 和 'KNN'，'GA+OP'
    Memory = 1024                # 内存的容量
    Delta = 32                   # 自适应k的更新间隔
    w = np.ones((N)) # [1.5 if i%2==0 else 1 for i in range(N)] # weights for each user
    # V = 17.6 # 17.7
    V = 10 # 10
    CHFACT = 1000000
    Local_A = 0.7
    Edge_A = 0.9
    w_fac = 0.5
    AR = 0 # 对精度的要求,小数，浮点数
 
    print('#user = %d, #channel number=%d, K=%d, decoder = %s, Memory = %d, Delta = %d, V = %d, w_fac = %f'%(N,n,K,decoder_mode, Memory, Delta, V, w_fac))
    # 导入数据
    # channel = sio.loadmat('./data/data_%d' %N)['input_h']
    # rate = sio.loadmat('./data/data_%d' %N)['output_obj'] # 这个数据仅用于画图，未用户训练模型.
    if new_flag == False:
        channel = sio.loadmat('./result_%d_3M' %N)['input_h']
        dataA = sio.loadmat('./result_%d_3M' %N)['data_arrival']
        # 将h增加到接近1以获得更好的训练；这是深度学习中广泛采用的技巧
        channel = channel * CHFACT
        plot_rate(channel.sum(axis=1)/N, 50, "channel gain")
        plot_rate(dataA.sum(axis=1)/N, 50, "dataA")
        if CD_gain_flag == False: # 使用原有的CD值
            # CD_gain = sio.loadmat('./result_CD_gain_%d_3M_DROO' %N)['CD_gain']
            CD_gain = sio.loadmat('./result_CD_gain_%d_3M' %N)['CD_gain']
            plot_rate(CD_gain[0], 50, "CD_gain")
        # np,show()
    else:
        # 尝试产生数据
        dataA = np.zeros((n,N))  # arrival data size
        channel = np.zeros((n,N)) # chanel gains
        arrival_lambda = 3*np.ones((N)) # average data arrival, 3 Mbps per user
    accuracy_req = np.zeros((n,N))  # 精度需求
    # generate channel
    # dist_v = np.linspace(start = 120, stop = 255, num = N) # AP与WD的距离
    dist_v = np.linspace(start = 2.5, stop = 5.2, num = N) # AP与WD的距离
    Ad = 3
    # Ad = 4.11
    fc = 915*10**6
    loss_exponent = 3 # path loss exponent
    # loss_exponent = 2.8 # dc
    light = 3*10**8
    h0 = np.ones((N))
    for j in range(0,N):
        h0[j] = Ad*(light/4/math.pi/fc/dist_v[j])**(loss_exponent)


    # 生成训练和测试数据样本索引
    # 数据被分割为 80:20
    # 如果 n > 总数据大小，则训练数据随机抽样，重复
    '''
    split_idx = int(.8 * len(channel)) # 训练集数量
    num_test = min(len(channel) - split_idx, n - int(.8 * n)) # 测试集数量
    '''
    # DNN神经网络初始化
    mem = MemoryDNN(net = [N*2, 120, 80, N],
                    # learning_rate = 0.01,
                    learning_rate = 0.0009,
                    training_interval=5,
                    batch_size=256,
                    memory_size=Memory
                    )
    alpha_mem = alpha_MemoryDNN(net = [N*3, 120, 80, 1],
                    #learning_rate = 0.01,
                    #training_interval=10,
                    #batch_size=128,
                    #memory_size=Memory
                    #)
                    learning_rate = 0.0009,
                    training_interval=5,
                    batch_size=256,
                    memory_size=Memory
                    )
    tau_mem = tau_MemoryDNN(net = [N*3, 120, 80, 1 + N],
                    # learning_rate = 0.0009,
                    # learning_rate = 0.0009,
                    # training_interval=5,
                    # batch_size=256,
                    learning_rate = 0.000001,
                    training_interval=5,
                    batch_size=256,
                    memory_size=Memory
                    )

    start_time = time.time()

    rate_his = []
    gain_his = []
    rate_his_ratio = []
    mode_his = []
    k_idx_his = []
    K_his = []
    Obj = []
    Obj1 = []
    Q = np.zeros((n,N)) # data queue in MbitsW
    rate_temp = np.zeros((n,N)) # achieved computation rate
    accuracy_sum = []
    alpha_list = []
    tau_list = []
    if CD_gain_flag == True: # 使用更新的CD值
        CD_gain = []
    
    for i in range(n):
        if i % (n//10) == 0:
           print("%0.1f"%(i/n))
        
        i_idx = i
        if new_flag == True:
            # 产生数据
            dataA[i_idx,:] = np.random.exponential(arrival_lambda)
            
            # 产生的信道
            h_tmp = racian_mec(h0, 0.3)
            # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
            h = h_tmp*CHFACT
        
            channel[i_idx,:] = h
        #accuracy_req[i_idx,:] = np.random.uniform(0.7, 0.8, N)  # 精度需求矩阵
        #AR = accuracy_req[i_idx,:]
        # 4) ‘Queueing module’ of LyDROO
        if i_idx > 0:
            # update queues
            # print("Q[i_idx-1, :] =", Q[i_idx-1, :])
            # print("dataA[i_idx-1, :] =", dataA[i_idx-1, :])
            # print("rate_temp[i_idx-1, :] =", rate_temp[i_idx-1, :])
            # print("mode_his[-1] =", mode_his[-1])
            Q[i_idx,:] = Q[i_idx-1,:] + dataA[i_idx-1,:] - rate_temp[i_idx-1,:] # current data queue
            # assert Q is positive due to float error
            Q[i_idx,Q[i_idx,:]<0] =0        
            # if i_idx == 5:
                # np,show()
        # print("Q[i_idx, :] =", Q[i_idx, :])
        
        h = channel[i_idx,:]
        # print("h =", h)
        # np,show()
        nn_input = np.concatenate((h, Q[i_idx,:]/10000)) #, Q[i_idx, :])) #, timefeature))

        # 卸载动作的生成方式选择 'OP' or 'KNN' or 'OPN'
        m_list = mem.decode(nn_input, K, decoder_mode)
        m_list_temp = [] # 真正满足精度的队列
        r_list_temp = []
        r_list_temp_alpha = []
        r_list = []
        
        # r_list_1 = []
        # r_list_temp_1 = []
        # m_1 = []
        # accuracy1 = []
        # 当前alpha
        alpha_list_local = []
        # 当前tau
        tau_list_local = []

        for m in m_list:
            # print("m =", m)
            # alpha预测
            #alpha_input = np.concatenate((h, Q[i_idx,:]/10000,m))
            #alpha = alpha_mem.decode(alpha_input) # 用于充电的时间
            #tau_sum = 1 - alpha # 用于卸载的时间总和
            # np,show(0)
            # tau预测
            tau_input = np.concatenate((h, Q[i_idx,:]/10000, m))
            tau = tau_mem.decode(tau_input)
            #print("tau =", tau)
            tau[1:] = tau[1:]*m
            tau = tau/sum(tau)
            #print("tau =", tau, sum(tau))
            #np,show()
            # if sum(m) != 0: # 至少存在一个卸载，那必然需要传输时间，存在tau
            #    tau_p = (m*tau)/sum(m*tau) # 对于不卸载的节点，概率为0
            #    tau_time = tau_p * tau_sum # 卸载节点分配的时间
            # else: # alpha.;.不存在卸载，不存在tau
            #    alpha = [1.0] # 说明节点不卸载任务，仅仅在本地处理，T时间内都可以充电
            #    tau_time = []
            #    for tua_i in range(len(tau)):
            #        tau_time.append(0)
            #    tau_time = np.array(tau_time)
                # print("tau_list =", tau_list)
            alpha_tau = [] # alpha和tau值组合好，放入网络训练
            alpha_tau.append(tau[0])
            # 充电时间和卸载时间的组合，构成一个列表
            for alpha_tau_i in range(1, len(tau)):
                if tau[alpha_tau_i] != 0:
                    alpha_tau.append(tau[alpha_tau_i])
            
            M0=np.where(m==0)[0]  # 取出为0的位置，进行本地计算
            M1=np.where(m==1)[0]  # 取出为1的位置，进行边缘卸载计算
           
            if len(M0)*Local_A + len(M1)*Edge_A >= len(m)*(AR): # 满足精度条件
                # ************************************************************
                # 用神经网络的方式求解
                r_list_temp_alpha.append(alpha_Mybisection(h, m, Q[i_idx, :], w, V, dataA[i_idx,:], Local_A, Edge_A, w_fac, alpha_tau))
                r_list_temp.append(r_list_temp_alpha[-1])    
                
                # **********************************************************
                # 用数学推理的方案求解：卸载动作，充电时间和传输时间
                # r_list_temp.append(Mybisection(h, m, Q[i_idx, :], w, V, dataA[i_idx,:], Local_A, Edge_A, w_fac))
                # **********************************************************

                # 实现综合考虑两种方案的方式
                if (np.random.rand() < 0.6):
                    r_list_temp_alpha.append(Mybisection(h, m, Q[i_idx, :], w, V, dataA[i_idx,:], Local_A, Edge_A, w_fac))
                    r_list_temp[-1] = r_list_temp_alpha[-1]
                
                m_list_temp.append(m) # 真正满足精度要求的队列

                # 目标函数
                r_list.append(r_list_temp[-1][0])
                # alpha
                alpha_list_local = []
                alpha_list_local.append(r_list_temp[-1][1])
                # tau
                tau_local_temp = np.zeros((N)) # 初始化时间数组
                # print("r_list_temp[-1][2] =", r_list_temp[-1][2])
                # print("M1 =", M1)
                tau_local_temp[M1] = r_list_temp[-1][2]

                for tau_local_i in range(N):
                    alpha_list_local.append(tau_local_temp[tau_local_i])

                #print("tau_local_temp =", tau_local_temp)
                #print("alpha_list_local =", alpha_list_local)
                tau_list_local.append(alpha_list_local)
                #np,show()

                
           
        #if len(r_list) != 0:
            m_temp = m_list_temp[np.argmax(r_list)]
            M0=np.where(m_temp==0)[0]  # 取出为0的位置，进行本地计算
            M1=np.where(m_temp==1)[0]  # 取出为1的位置，进行边缘卸载计算
            accuracy_sum.append(len(M0)*Local_A + len(M1)*Edge_A)
        else: # 精度没有符合要求的情况下的操作, 咱们就用CD取得满足精度的卸载方式
            # **********************************************************
            # CD 方案求解：卸载动作，充电时间和传输时间
            # **********************************************************
            gain0, CD_a, CD_Tj, CD_rate, CD_m = alpha_Mycd_method(h, Q[i_idx, :], w, V, dataA[i_idx,:], Local_A, Edge_A, w_fac, alpha_tau, AR)
            CD_val = []
            CD_val.append(gain0)
            CD_val.append(CD_a)
            CD_val.append(CD_Tj)
            CD_val.append(CD_rate)
            # 尽可能生成符合精度条件的值
            r_list_temp.append(CD_val)
            
            # r_list_temp[-1][4]即为CD需要的卸载动作
            M0=np.where(CD_m==0)[0]  # 取出为0的位置，进行本地计算
            M1=np.where(CD_m==1)[0]  # 取出为1的位置，进行边缘卸载计算
           
            
            # m_list = [] # 当前生成的卸载动作都不能满足精度的要求，需要CD生成合理的卸载动作
            # r_list_temp = []
            # r_list = []
            m_list_temp.append(CD_m) # CD生成的卸载方式
            r_list.append(r_list_temp[-1][0])
            accuracy_sum.append(len(M0)*Local_A + len(M1)*Edge_A)

            # alpha
            alpha_list_local = []
            alpha_list_local.append(r_list_temp[-1][1])
            # tau
            tau_local_temp = np.zeros((N)) # 初始化时间数组
            # print("r_list_temp[-1][2] =", r_list_temp[-1][2])
            # print("M1 =", M1)
            tau_local_temp[M1] = r_list_temp[-1][2]

            for tau_local_i in range(N):
                alpha_list_local.append(tau_local_temp[tau_local_i])

            #print("tau_local_temp =", tau_local_temp)
            #print("alpha_list_local =", alpha_list_local)
            tau_list_local.append(alpha_list_local)
            

        # print("dataA[i_idx, :] =", dataA[i_idx, :])
        # np,show()
        # cd_method方案来自文献：“Computation rate maximization for wireless powered mobile-edge computing with binary computation ofﬂoading,”
        # gain0, M0 = cd_method(h/1000000)
        if CD_gain_flag == True: # 使用更新的CD值
            gain0, CD_a, CD_Tj, CD_rate, CD_m = alpha_Mycd_method(h, Q[i_idx, :], w, V, dataA[i_idx,:], Local_A, Edge_A, w_fac, alpha_tau, AR)
            CD_gain.append(gain0)
        # 对应奖励最大的卸载决策m与输入h存入内存，用于训练DNN网络
        # print("np.argmax(r_list) =", np.argmax(r_list))
        # np,show()
        mem.encode(nn_input, m_list_temp[np.argmax(r_list)])
        #alpha_mem.encode(alpha_input,alpha_list_local[np.argmax(r_list)])
        tau_mem.encode(tau_input,tau_list_local[np.argmax(r_list)])
        # print(tau_list[np.argmax(r_list)])
        # print(tau_list)
        # np,show()
        # DROO的主要代码到此为止



        # 以下代码存储了一些感兴趣的指标用于论文画图用
        # 存储最大奖励
        tau_list.append(tau_list_local[np.argmax(r_list)])
        #alpha_list.append(alpha_list_local[np.argmax(r_list)])
        rate_his.append(np.max(r_list))
        if CD_gain_flag == False: # 使用原有的CD值
            gain_his.append(CD_gain[0][i_idx])
        else:
            gain_his.append(gain0)
        rate_his_ratio.append(rate_his[-1] / gain_his[-1]) # / rate[i_idx][0]) # 公式11，参考：Deep Reinforcement Learning for Online Ofﬂoading in Wireless Powered Mobile-Edge Computing Networks
        # 其中rate[i_idx][0]的数据来自cd方案，也可以用gain0表示，存在：gain0 = rate[i_idx][0]
        # 记录最大奖励的索引
        k_idx_his.append(np.argmax(r_list))
        # 在自适应k的情况下记录k
        K_his.append(K)
        mode_his.append(m_list_temp[np.argmax(r_list)])

        x, y, z, rate_temp[i_idx, :] = r_list_temp[k_idx_his[-1]]
        # print("m_list[np.argmax(r_list)] =", m_list[np.argmax(r_list)])
        # print("x =", x)
        # print("y =", y)
        # print("z =", z)
        
         


    total_time=time.time()-start_time
    mem.plot_cost()
    alpha_mem.plot_cost()
    tau_mem.plot_cost()
    # print("dataA[0:, 0] =", dataA[0:, 0])
    plot_rate(rate_his_ratio)
    num_i = []
    for n_i in range(n):
        num_i.append(n_i)
    import matplotlib.pyplot as plt
    #plt.plot(num_i, rate_his_ratio)
    #plt.show()
    #plt.plot(num_i, rate_his)
    #plt.show()
    plot_rate(rate_his, 50, "QDRL_RCR")
    plot_rate(gain_his, 50, "CD_RCR")
    # print("tau_list =", tau_list, len(tau_list))
    #np,show()
    plot_rate(np.array(tau_list).sum(axis=1)/N, 50, "tau_list")
    #plot_rate(alpha_list, 50, "alpha")
    #print("alpha_list =", alpha_list, len(alpha_list))
    plt.plot(num_i, np.array(tau_list).sum(axis=1)/N)
    plt.show()
    #plt.plot(num_i, alpha_list)
    plt.show()
    #np,show()
    # 队列
    plot_rate(Q.sum(axis=1)/N, 50, "Queue")
    # 计算率
    plot_rate(rate_temp.sum(axis=1)/N, 50, "rate_temp")
    # 精确度
    plot_rate(accuracy_sum, 50, "accuracy")
    
    # 多条队列的动态变化
    #plot_rate(Q[0:, 0], 50, "1 queue")
    #plot_rate(Q[0:, 1], 50, "2 queue")
    #plot_rate(Q[0:, 2], 50, "3 queue")
    #plot_rate(Q[0:, 3], 50, "4 queue")
    #plot_rate(Q[0:, 4], 50, "5 queue")
    #plot_rate(Q[0:, 5], 50, "6 queue")
    #plot_rate(Q[0:, 6], 50, "7 queue")
    #plot_rate(Q[0:, 7], 50, "8 queue")
    #plot_rate(Q[0:, 8], 50, "9 queue")
    #plot_rate(Q[0:, 9], 50, "10 queue")

    #plot_rate((channel).sum(axis=1)/N, 50, "one queue")
    #plot_rate((channel/CHFACT).sum(axis=1)/N, 50, "one queue")
    #plot_rate(dataA.sum(axis=1)/N, 50, "one queue")
    
    # print("Averaged normalized computation rate:", sum(rate_his_ratio[-num_test: -1])/num_test)
    print("Averaged normalized computation rate:", sum(rate_his_ratio[0:])/n)
    print('Total time consumed:%s'%total_time)
    print('Average time per channel:%s'%(total_time/n))
    # np,show()
    # save data into txt
    save_to_txt(k_idx_his, "k_idx_his.txt")
    save_to_txt(K_his, "K_his.txt")
    save_to_txt(mem.cost_his, "save_cost_his.txt") # loss随迭代次数的变化
    save_to_txt(tau_mem.cost_his, "save_tau_mem_cost_his.txt") # loss随迭代次数的变化
    save_to_txt(rate_his_ratio, "rate_his_ratio.txt")
    save_to_txt(mode_his, "mode_his.txt")
    save_to_txt(Q.sum(axis=1)/N, "save_Queue.txt") # 队列变化
    save_to_txt(rate_his, "save_rate_his.txt") # 真实计算率
    save_to_txt(accuracy_sum, "save_accuracy_sum.txt") # 精确度
    save_to_txt(rate_temp.sum(axis=1)/N, "save_sensor_rate_his") # 每个设备的计算率
    print('#user = %d, #channel number=%d, K=%d, decoder = %s, Memory = %d, Delta = %d, V = %d, w_fac = %f'%(N,n,K,decoder_mode, Memory, Delta, V, w_fac))
    print("ave queue:", sum(Q.sum(axis=1)/N)/n)# 平均队列
    print("ave rate_his:", sum(rate_his)/n) # 平均真实计算率
    print("ave accuracy_sum:", sum(accuracy_sum)/n) # 平均精确度
    print("ave rate_temp:", sum(rate_temp.sum(axis=1)/N)/n)#设备平均计算率
    print("ave rate_his_ratio:", sum(rate_his_ratio)/n) #比例
    if CD_gain_flag == True: # 保存更新的CD值
        sio.savemat('./result_CD_gain_%d_3M.mat'%N, {'CD_gain': CD_gain})
    if new_flag == True:
        plot_rate(channel.sum(axis=1)/N, 50, "channel gain")
        plot_rate(dataA.sum(axis=1)/N, 50, "dataA")
        # save all data
        sio.savemat('./result_%d.mat'%N, {'input_h': channel/CHFACT,'data_arrival':dataA,'data_queue':Q,'off_mode':mode_his,'rate':rate_temp,'objective':rate_his_ratio})
    