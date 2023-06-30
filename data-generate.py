from re import T
import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy
import math
 
from memoryPyTorch import MemoryDNN 
from optimization import bisection, cd_method, Mybisection, Mycd_method

import time
def plot_rate(rate_his, rolling_intv=50, ylabel = 'Normalized Computation Rate'): 
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
        input_h:state
        r_list:reward
        m_list:action
    '''
    new_flag = True 
    N = 10                       
    n = 30000                     
    K = N                        
    decoder_mode = 'OP'          
    Memory = 1024                
    Delta = 32                   
    w = np.ones((N)) # [1.5 if i%2==0 else 1 for i in range(N)] # weights for each user
    V = 20
    DataArrival = 2.9
    CHFACT = 1000000
 
    print('#user = %d, #channel number=%d, K=%d, decoder = %s, Memory = %d, Delta = %d'%(N,n,K,decoder_mode, Memory, Delta))
    if new_flag == False:
        channel = sio.loadmat('./result_%d' %N)['input_h']
        dataA = sio.loadmat('./result_%d' %N)['data_arrival']
        channel = channel * CHFACT
        plot_rate(channel.sum(axis=1)/N, 50, "channel gain")
        plot_rate(dataA.sum(axis=1)/N, 50, "dataA")
    else:
        dataA = np.zeros((n,N))  # arrival data size
        channel = np.zeros((n,N)) # chanel gains
        arrival_lambda = DataArrival*np.ones((N)) # average data arrival, 3 Mbps per user

    # generate channel
    dist_v = np.linspace(start = 2.5, stop = 5.2, num = N) 
    Ad = 3
    # Ad = 4.11
    fc = 915*10**6
    loss_exponent = 3 # path loss exponent
    # loss_exponent = 2.8 # dc
    light = 3*10**8
    h0 = np.ones((N))
    for j in range(0,N):
        h0[j] = Ad*(light/4/math.pi/fc/dist_v[j])**(loss_exponent)

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
            dataA[i_idx,:] = np.random.exponential(arrival_lambda)
            h_tmp = racian_mec(h0, 0.3)
            # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
            h = h_tmp*CHFACT
        
            channel[i_idx,:] = h

    if new_flag == True:
        plot_rate(channel.sum(axis=1)/N, 50, "channel gain")
        plot_rate(dataA.sum(axis=1)/N, 50, "dataA")
        # save all data
        sio.savemat('./result_%d_2.9M.mat'%N, {'input_h': channel/CHFACT,'data_arrival':dataA,'data_queue':Q,'off_mode':mode_his,'rate':rate_temp,'objective':rate_his_ratio})
