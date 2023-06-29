from re import T
import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy
import math

# 基于PyTorch实现 
from memoryPyTorch import MemoryDNN # 做卸载决策所使用的神经网络模型（DNN），包含推理、训练、存储（输入，标签）对和采样数据
from optimization import bisection, cd_method, Mybisection, Mycd_method # 资源分配

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
    new_flag = True # 是否需要新产生数据，False是不需要产生新数据
    N = 10                       # 用户数量
    n = 30000                     # 时间帧的数量（决定了整个系统运行多久）
    K = N                        # 初始化 K = N
    decoder_mode = 'OP'          # 量化卸载决策的方式可选 'OP' (Order-preserving) 和 'KNN'
    Memory = 1024                # 内存的容量
    Delta = 32                   # 自适应k的更新间隔
    w = np.ones((N)) # [1.5 if i%2==0 else 1 for i in range(N)] # weights for each user
    V = 20
    DataArrival = 2.9
    CHFACT = 1000000
 
    print('#user = %d, #channel number=%d, K=%d, decoder = %s, Memory = %d, Delta = %d'%(N,n,K,decoder_mode, Memory, Delta))
    # 导入数据
    # channel = sio.loadmat('./data/data_%d' %N)['input_h']
    # rate = sio.loadmat('./data/data_%d' %N)['output_obj'] # 这个数据仅用于画图，未用户训练模型.
    if new_flag == False:
        channel = sio.loadmat('./result_%d' %N)['input_h']
        dataA = sio.loadmat('./result_%d' %N)['data_arrival']
        # 将h增加到接近1以获得更好的训练；这是深度学习中广泛采用的技巧
        channel = channel * CHFACT
        plot_rate(channel.sum(axis=1)/N, 50, "channel gain")
        plot_rate(dataA.sum(axis=1)/N, 50, "dataA")
        # np,show()
    else:
        # 尝试产生数据
        dataA = np.zeros((n,N))  # arrival data size
        channel = np.zeros((n,N)) # chanel gains
        arrival_lambda = DataArrival*np.ones((N)) # average data arrival, 3 Mbps per user

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
                    learning_rate = 0.01,
                    training_interval=10,
                    batch_size=128,
                    memory_size=Memory
                    )

    start_time = time.time()

    rate_his = []
    rate_his_ratio = []
    mode_his = []
    k_idx_his = []
    K_his = []
    Obj = []
    Obj1 = []
    Q = np.zeros((n,N)) # data queue in MbitsW
    rate_temp = np.zeros((n,N)) # achieved computation rate
    for i in range(n):
        
        i_idx = i
        if new_flag == True:
            # 产生数据
            dataA[i_idx,:] = np.random.exponential(arrival_lambda)
            # 产生的信道
            h_tmp = racian_mec(h0, 0.3)
            # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
            h = h_tmp*CHFACT
        
            channel[i_idx,:] = h

    if new_flag == True:
        plot_rate(channel.sum(axis=1)/N, 50, "channel gain")
        plot_rate(dataA.sum(axis=1)/N, 50, "dataA")
        # save all data
        sio.savemat('./result_%d_2.9M.mat'%N, {'input_h': channel/CHFACT,'data_arrival':dataA,'data_queue':Q,'off_mode':mode_his,'rate':rate_temp,'objective':rate_his_ratio})
