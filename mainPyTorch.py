
import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy
import math

from memoryPyTorch import MemoryDNN, alpha_MemoryDNN, tau_MemoryDNN 
from optimization import bisection, cd_method, Mybisection, Mycd_method, alpha_Mybisection, alpha_Mycd_method 

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
    new_flag = False 
    CD_gain_flag = False  
    N = 10                       
    n = 500                    
    K = N                        
    decoder_mode = 'GA+OP'         
    Memory = 1024                
    Delta = 32                   
    w = np.ones((N)) # [1.5 if i%2==0 else 1 for i in range(N)] # weights for each user
    # V = 17.6 # 17.7
    V = 10 # 10
    CHFACT = 1000000
    Local_A = 0.7
    Edge_A = 0.9
    w_fac = 0.5
    AR = 0 
 
    print('#user = %d, #channel number=%d, K=%d, decoder = %s, Memory = %d, Delta = %d, V = %d, w_fac = %f'%(N,n,K,decoder_mode, Memory, Delta, V, w_fac))
    if new_flag == False:
        channel = sio.loadmat('./result_%d_3M' %N)['input_h']
        dataA = sio.loadmat('./result_%d_3M' %N)['data_arrival']

        channel = channel * CHFACT
        plot_rate(channel.sum(axis=1)/N, 50, "channel gain")
        plot_rate(dataA.sum(axis=1)/N, 50, "dataA")
        if CD_gain_flag == False: 
            CD_gain = sio.loadmat('./result_CD_gain_%d_3M' %N)['CD_gain']
            plot_rate(CD_gain[0], 50, "CD_gain")
    else:
        dataA = np.zeros((n,N))  # arrival data size
        channel = np.zeros((n,N)) # chanel gains
        arrival_lambda = 3*np.ones((N)) # average data arrival, 3 Mbps per user
    accuracy_req = np.zeros((n,N))  
    # generate channel
    dist_v = np.linspace(start = 2.5, stop = 5.2, num = N) 
    Ad = 3
    # Ad = 4.11
    fc = 915*10**6
    loss_exponent = 3 # path loss exponent
    light = 3*10**8
    h0 = np.ones((N))
    for j in range(0,N):
        h0[j] = Ad*(light/4/math.pi/fc/dist_v[j])**(loss_exponent)

    # DNN
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
    if CD_gain_flag == True: 
        CD_gain = []
    
    for i in range(n):
        if i % (n//10) == 0:
           print("%0.1f"%(i/n))
        
        i_idx = i
        if new_flag == True:
            dataA[i_idx,:] = np.random.exponential(arrival_lambda)
            h_tmp = racian_mec(h0, 0.3)
            # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
            h = h_tmp*CHFACT
        
            channel[i_idx,:] = h

        # 4) ‘Queueing module’ of QDRL
        if i_idx > 0:
            Q[i_idx,:] = Q[i_idx-1,:] + dataA[i_idx-1,:] - rate_temp[i_idx-1,:] # current data queue
            Q[i_idx,Q[i_idx,:]<0] =0        

        h = channel[i_idx,:]
        nn_input = np.concatenate((h, Q[i_idx,:]/10000)) #, Q[i_idx, :])) #, timefeature))
        m_list = mem.decode(nn_input, K, decoder_mode)
        m_list_temp = []
        r_list_temp = []
        r_list_temp_alpha = []
        r_list = []
        
        alpha_list_local = []
        tau_list_local = []

        for m in m_list:
            tau_input = np.concatenate((h, Q[i_idx,:]/10000, m))
            tau = tau_mem.decode(tau_input)
            tau[1:] = tau[1:]*m
            tau = tau/sum(tau)
            alpha_tau = [] 
            alpha_tau.append(tau[0])
            for alpha_tau_i in range(1, len(tau)):
                if tau[alpha_tau_i] != 0:
                    alpha_tau.append(tau[alpha_tau_i])
            
            M0=np.where(m==0)[0]  
            M1=np.where(m==1)[0]  
           
            if len(M0)*Local_A + len(M1)*Edge_A >= len(m)*(AR): 
                r_list_temp_alpha.append(alpha_Mybisection(h, m, Q[i_idx, :], w, V, dataA[i_idx,:], Local_A, Edge_A, w_fac, alpha_tau))
                r_list_temp.append(r_list_temp_alpha[-1])    
    
                if (np.random.rand() < 0.6):
                    r_list_temp_alpha.append(Mybisection(h, m, Q[i_idx, :], w, V, dataA[i_idx,:], Local_A, Edge_A, w_fac))
                    r_list_temp[-1] = r_list_temp_alpha[-1]
                
                m_list_temp.append(m) 

                # rewrad list
                r_list.append(r_list_temp[-1][0])
                # alpha
                alpha_list_local = []
                alpha_list_local.append(r_list_temp[-1][1])
                # tau
                tau_local_temp = np.zeros((N)) 
                tau_local_temp[M1] = r_list_temp[-1][2]

                for tau_local_i in range(N):
                    alpha_list_local.append(tau_local_temp[tau_local_i])
                tau_list_local.append(alpha_list_local)

            m_temp = m_list_temp[np.argmax(r_list)]
            M0=np.where(m_temp==0)[0]  
            M1=np.where(m_temp==1)[0]  
            accuracy_sum.append(len(M0)*Local_A + len(M1)*Edge_A)
        else: # 
            # CD
            gain0, CD_a, CD_Tj, CD_rate, CD_m = alpha_Mycd_method(h, Q[i_idx, :], w, V, dataA[i_idx,:], Local_A, Edge_A, w_fac, alpha_tau, AR)
            CD_val = []
            CD_val.append(gain0)
            CD_val.append(CD_a)
            CD_val.append(CD_Tj)
            CD_val.append(CD_rate)
            r_list_temp.append(CD_val)
            
            M0=np.where(CD_m==0)[0] 
            M1=np.where(CD_m==1)[0]  
           
            m_list_temp.append(CD_m) 
            r_list.append(r_list_temp[-1][0])
            accuracy_sum.append(len(M0)*Local_A + len(M1)*Edge_A)

            # alpha
            alpha_list_local = []
            alpha_list_local.append(r_list_temp[-1][1])
            # tau
            tau_local_temp = np.zeros((N)) 
            tau_local_temp[M1] = r_list_temp[-1][2]

            for tau_local_i in range(N):
                alpha_list_local.append(tau_local_temp[tau_local_i])
            tau_list_local.append(alpha_list_local)
            
        if CD_gain_flag == True: 
            gain0, CD_a, CD_Tj, CD_rate, CD_m = alpha_Mycd_method(h, Q[i_idx, :], w, V, dataA[i_idx,:], Local_A, Edge_A, w_fac, alpha_tau, AR)
            CD_gain.append(gain0)
        mem.encode(nn_input, m_list_temp[np.argmax(r_list)])
       
        tau_mem.encode(tau_input,tau_list_local[np.argmax(r_list)])
        
        tau_list.append(tau_list_local[np.argmax(r_list)])
        rate_his.append(np.max(r_list))
        if CD_gain_flag == False: #
            gain_his.append(CD_gain[0][i_idx])
        else:
            gain_his.append(gain0)
        rate_his_ratio.append(rate_his[-1] / gain_his[-1]) # / rate[i_idx][0])
        
        k_idx_his.append(np.argmax(r_list))
        K_his.append(K)
        mode_his.append(m_list_temp[np.argmax(r_list)])

        x, y, z, rate_temp[i_idx, :] = r_list_temp[k_idx_his[-1]]
        
    total_time=time.time()-start_time
    mem.plot_cost()
    alpha_mem.plot_cost()
    tau_mem.plot_cost()
    plot_rate(rate_his_ratio)
    num_i = []
    for n_i in range(n):
        num_i.append(n_i)
    import matplotlib.pyplot as plt
    plot_rate(rate_his, 50, "QDRL_RCR")
    plot_rate(gain_his, 50, "CD_RCR")
    plot_rate(np.array(tau_list).sum(axis=1)/N, 50, "tau_list")
    
    plt.plot(num_i, np.array(tau_list).sum(axis=1)/N)
    plt.show()
    plot_rate(Q.sum(axis=1)/N, 50, "Queue")
    plot_rate(rate_temp.sum(axis=1)/N, 50, "rate_temp")
    plot_rate(accuracy_sum, 50, "accuracy")
    
    print("Averaged normalized computation rate:", sum(rate_his_ratio[0:])/n)
    print('Total time consumed:%s'%total_time)
    print('Average time per channel:%s'%(total_time/n))
    # save data into txt
    save_to_txt(k_idx_his, "k_idx_his.txt")
    save_to_txt(K_his, "K_his.txt")
    save_to_txt(mem.cost_his, "save_cost_his.txt") # los
    save_to_txt(tau_mem.cost_his, "save_tau_mem_cost_his.txt") # loss
    save_to_txt(rate_his_ratio, "rate_his_ratio.txt")
    save_to_txt(mode_his, "mode_his.txt")
    save_to_txt(Q.sum(axis=1)/N, "save_Queue.txt") # 
    save_to_txt(rate_his, "save_rate_his.txt") # 
    save_to_txt(accuracy_sum, "save_accuracy_sum.txt") #
    save_to_txt(rate_temp.sum(axis=1)/N, "save_sensor_rate_his") # 
    print('#user = %d, #channel number=%d, K=%d, decoder = %s, Memory = %d, Delta = %d, V = %d, w_fac = %f'%(N,n,K,decoder_mode, Memory, Delta, V, w_fac))
    print("ave queue:", sum(Q.sum(axis=1)/N)/n)# 
    print("ave rate_his:", sum(rate_his)/n) # 
    print("ave accuracy_sum:", sum(accuracy_sum)/n) # 
    print("ave rate_temp:", sum(rate_temp.sum(axis=1)/N)/n)#
    print("ave rate_his_ratio:", sum(rate_his_ratio)/n) #
    if CD_gain_flag == True: # 
        sio.savemat('./result_CD_gain_%d_3M.mat'%N, {'CD_gain': CD_gain})
    if new_flag == True:
        plot_rate(channel.sum(axis=1)/N, 50, "channel gain")
        plot_rate(dataA.sum(axis=1)/N, 50, "dataA")
        # save all data
        sio.savemat('./result_%d.mat'%N, {'input_h': channel/CHFACT,'data_arrival':dataA,'data_queue':Q,'off_mode':mode_his,'rate':rate_temp,'objective':rate_his_ratio})
    
