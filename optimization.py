# -*- coding: utf-8 -*-
#  ################################################################
#  该文件包含：计算资源的优化分配
#  ###################################################################

import numpy as np
from scipy import optimize
from scipy.special import lambertw
import scipy.io as sio                     
import time


def plot_gain( gain_his): # 画图
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl
    
    gain_array = np.asarray(gain_his)
    df = pd.DataFrame(gain_his)
    
    
    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15,8))
    rolling_intv = 20

    plt.plot(np.arange(len(gain_array))+1, df.rolling(rolling_intv, min_periods=1).mean(), 'b')
    plt.fill_between(np.arange(len(gain_array))+1, df.rolling(rolling_intv, min_periods=1).min()[0], df.rolling(rolling_intv, min_periods=1).max()[0], color = 'b', alpha = 0.2)
    plt.ylabel('Gain ratio')
    plt.xlabel('learning steps')
    plt.show()

def alpha_Mybisection(h, M, Q, weights, V, dataA, Local_A, Edge_A, w_fac, alpha_tau):
    # the bisection algorithm proposed by Suzhi BI
    # average time to find the optimal: 0.012535839796066284 s

    # parameters and equations
    # 公式1的参数解释
    o=100  # 处理1bit任务数据所需周期数
    # p=3  # AP在发射射频能量时的传输功率
    p = 3 # W
    u = 0.7 # 能量收集率
    eta1=((u*p)**(1.0/3))/o # -->本地计算中的n1-->固定参数
    # 可选参数
    d_fact = 10**6
    # d_fact = 10**5
    # d_fact = 10**4
    # d_fact = 1024*8
    # ki=(10**-26)*(d_fact**3)   # 计算能量效率系数
    ki=(10**-26)   # 计算能量效率系数
    
    ch_fact = 10**10
    # ki=(10**(-26))*(d_fact**3)   # 计算能量效率系数
    
    # Local_A =  0.7 # 0.7 # 设备端模型的精度
    # Edge_A =  0.9 # 0.9 # 边缘服务器上模型的精度
    # error_fac = 3
    # w_fac = 0.5
    # 公式3的参数解释
    # u为能量收集效率，p为AP在发射射频能量时的传输功率
    
    B=2*10**6 # 带宽
    eta2=u*p/(10**(-10)) # u*p/N0  -->10**-10为接受噪音功率即N0
    # N0 = B*(10**(-17.4))*(10**(-3))*ch_fact/10**6 # noise power in watt
    # eta2=u*p/N0
    Vu=1.1 # 通信开销
    epsilon=B/(Vu*np.log(2)) # 带宽/通信开销 -->log2(x) = 1/ln2 * ln(x)
    x = [] # a =x[0], and tau_j = a[1:]
    
    M0=np.where(M==0)[0]  # 取出为0的位置，进行本地计算
    M1=np.where(M==1)[0]  # 取出为1的位置，进行边缘卸载计算

    # print("M0 =", M0)
    # print("M1 =", M1)
    # np,show()
    
    hi=np.array([h[i] for i in M0])  # 信道选择
    hj=np.array([h[i] for i in M1])  # 信道选择
    # print("M =", M)
    # print("dataA =", dataA)
    dataAi=np.array([dataA[i] for i in M0])  # 数据选择
    dataAj=np.array([dataA[i] for i in M1])  # 数据选择
    # print("dataAi =", dataAi)
    # print("dataAj =", dataAj)
    # np,show()
    if len(weights) == 0:
        # default weights [1, 1.5, 1, 1.5, 1, 1.5, ...]
        # weights = [1.5 if i%2==1 else 1 for i in range(len(M))]
        weights = np.ones((N));
    
    N = len(Q)
    
    a =  np.ones((N)) # control parameter in (26) of paper
    for i in range(len(a)):
        a[i] = Q[i]  + V*weights[i]
        # a[i] = weights[i]


    wi=np.array([weights[M0[i]] for i in range(len(M0))])
    wj=np.array([weights[M1[i]] for i in range(len(M1))])
    
    ai=np.array([a[M0[i]] for i in range(len(M0))])
    aj=np.array([a[M1[i]] for i in range(len(M1))])

    # print("ai =", ai)
    # print("aj =", aj)
    # np,show(0)
    rate = np.zeros((N));
    error = np.zeros((N));
    def sum_rate(x): # x[0]就是公式里面的a，它为AP广播射频能量共给WD收集的时间百分比，就是给WD充电的时间
        for i in range(len(M0)):
            temp_id = M0[i]
            # print("wi[i] =", wi[i])
            # print("wi[i]*eta1*(hi/ki)**(1.0/3)*x[0]**(1.0/3) =", wi[i]*eta1*(hi[i]/ki)**(1.0/3)*x[0]**(1.0/3))
            rate[temp_id] = (wi[i]*eta1*(hi[i]/ki)**(1.0/3)*x[0]**(1.0/3))/(d_fact)
            # print("Local_A =", Local_A)
            # print("dataAi[i] =", dataAi[i])
            error[temp_id] = (1.0 - (Local_A))*dataAi[i]# *ai[i]
            # print("error[temp_id] =", error[temp_id])
            # np,show()
        sum1=sum(ai*eta1*(hi/ki)**(1.0/3)*x[0]**(1.0/3)) # 本地计算的计算率总和，wi是节点权重，eta1=((u*p)**(1.0/3))/o
        # print("sum1 =", sum1)
        # print("rate =", rate)
        # np,show()
        sum2=0
        
        for i in range(len(M1)): # x[1:]，为分配给第i个wd用于任务卸载的时间比，
            temp_id = M1[i]
            rate[temp_id] = (wj[i]*epsilon*x[i+1]*np.log(1+eta2*hj[i]**2*x[0]/x[i+1]))/(d_fact)
            # print("dataAi[i] =", dataAi[i])
            error[temp_id] = (1- Edge_A)*dataAj[i] # *aj[i]
            # print("error[temp_id] =", error[temp_id])
            # np,show()
            sum2 += aj[i]*epsilon*x[i+1]*np.log(1+eta2*hj[i]**2*x[0]/x[i+1]) # 公式3，计算边缘计算的计算率
        return (sum1+sum2)/d_fact - V*w_fac*sum(error) # 加了个惩罚因子
    return sum_rate(alpha_tau), alpha_tau[0], alpha_tau[1:], rate

def alpha_Mycd_method(h, Q, weights, V, dataA, Local_A, Edge_A, w_fac, alpha_tau, AR):
    N = len(h) 
    M_origin = np.random.randint(2,size = N)
    M0=np.where(M_origin==0)[0]  # 取出为0的位置，进行本地计算
    M1=np.where(M_origin==1)[0]  # 取出为1的位置，进行边缘卸载计算
    while len(M0)*Local_A + len(M1)*Edge_A < len(M_origin)*(AR): # 一定要保证精度哦
        M_origin = np.random.randint(2,size = N)
        M0=np.where(M_origin==0)[0]  # 取出为0的位置，进行本地计算
        M1=np.where(M_origin==1)[0]  # 取出为1的位置，进行边缘卸载计算
    #print("M_origin =", M_origin)
    # gain0,a,Tj, _ = bisection(h,M0)
    gain0,a,Tj, _ = Mybisection(h, M_origin, Q, weights, V, dataA, Local_A, Edge_A, w_fac)
    #print("Tj =", Tj)
    g_list = []
    M_list = []
    g_list.append(gain0)
    M_list.append(M_origin)
    #a_list = []
    #Tj_list = []
    #a_list.append(a)
    #Tj_list.append(Tj)
    while True:
    # while len(M0)*Local_A + len(M1)*Edge_A < len(M)*(AR):
        for j in range(0,N):
            M = np.copy(M_origin)
            M[j] = (M[j]+1)%2
            # print("M =", M)
            M0=np.where(M==0)[0]  # 取出为0的位置，进行本地计算
            M1=np.where(M==1)[0]  # 取出为1的位置，进行边缘卸载计算
            # print(M0, M1)
            # np,show()
            # gain,a,Tj= bisection(h,M)
            if len(M0)*Local_A + len(M1)*Edge_A >= len(M)*(AR): # 满足精度条件
                #print("meet case j =", j)
                gain, a, Tj, _ = Mybisection(h, M, Q, weights, V, dataA, Local_A, Edge_A, w_fac)
                #print("gain, a, Tj, _ , M=",gain, a, Tj, _, M)
                g_list.append(gain)
                M_list.append(M)
                #a_list.append(a)
                #Tj_list.append(Tj)
        # print("what")
        g_max = max(g_list)
        #print("g_max =", g_max)
        if g_max > gain0:
            #print("g_list.index(g_max) =", g_list.index(g_max))
            gain0 = g_max
            M_origin = M_list[g_list.index(g_max)]
            #a = a_list[g_list.index(g_max)]
            # Tj = Tj_list[g_list.index(g_max)]

            #print("gain, a, Tj, _ , M=",gain0, a, Tj, _, M_origin)
        else:
            # print("heihei")
            break
    #print("M_origin =", M_origin)
    M = M_origin
    gain0, a, Tj, _ = Mybisection(h, M, Q, weights, V, dataA, Local_A, Edge_A, w_fac)
    #print("gain0, a, Tj, _, M = ", gain0, a, Tj, _, M)
    #np,show()
    return gain0, a, Tj, _, M

def Mybisection(h, M, Q, weights, V, dataA, Local_A, Edge_A, w_fac):
    # the bisection algorithm proposed by Suzhi BI
    # average time to find the optimal: 0.012535839796066284 s

    # parameters and equations
    # 公式1的参数解释
    o=100  # 处理1bit任务数据所需周期数
    # p=3  # AP在发射射频能量时的传输功率
    p = 3 # W
    u = 0.7 # 能量收集率
    eta1=((u*p)**(1.0/3))/o # -->本地计算中的n1-->固定参数
    # 可选参数
    d_fact = 10**6
    # d_fact = 10**5
    # d_fact = 10**4
    # d_fact = 1024*8
    # ki=(10**-26)*(d_fact**3)   # 计算能量效率系数
    ki=(10**-26)   # 计算能量效率系数
    
    ch_fact = 10**10
    # ki=(10**(-26))*(d_fact**3)   # 计算能量效率系数
    
    # Local_A =  0.7 # 0.7 # 设备端模型的精度
    # Edge_A =  0.9 # 0.9 # 边缘服务器上模型的精度
    # error_fac = 3
    # w_fac = 0.5
    # 公式3的参数解释
    # u为能量收集效率，p为AP在发射射频能量时的传输功率
    
    B=2*10**6 # 带宽
    eta2=u*p/(10**(-10)) # u*p/N0  -->10**-10为接受噪音功率即N0
    # N0 = B*(10**(-17.4))*(10**(-3))*ch_fact/10**6 # noise power in watt
    # eta2=u*p/N0
    Vu=1.1 # 通信开销
    epsilon=B/(Vu*np.log(2)) # 带宽/通信开销 -->log2(x) = 1/ln2 * ln(x)
    x = [] # a =x[0], and tau_j = a[1:]
    
    M0=np.where(M==0)[0]  # 取出为0的位置，进行本地计算
    M1=np.where(M==1)[0]  # 取出为1的位置，进行边缘卸载计算

    # print("M0 =", M0)
    # print("M1 =", M1)
    # np,show()
    
    hi=np.array([h[i] for i in M0])  # 信道选择
    hj=np.array([h[i] for i in M1])  # 信道选择
    # print("M =", M)
    # print("dataA =", dataA)
    dataAi=np.array([dataA[i] for i in M0])  # 数据选择
    dataAj=np.array([dataA[i] for i in M1])  # 数据选择
    # print("dataAi =", dataAi)
    # print("dataAj =", dataAj)
    # np,show()
    if len(weights) == 0:
        # default weights [1, 1.5, 1, 1.5, 1, 1.5, ...]
        # weights = [1.5 if i%2==1 else 1 for i in range(len(M))]
        weights = np.ones((N));
    
    N = len(Q)
    
    a =  np.ones((N)) # control parameter in (26) of paper
    for i in range(len(a)):
        # a[i] = Q[i]  + V*weights[i]
        a[i] = weights[i]

    wi=np.array([weights[M0[i]] for i in range(len(M0))])
    wj=np.array([weights[M1[i]] for i in range(len(M1))])
    
    ai=np.array([a[M0[i]] for i in range(len(M0))])
    aj=np.array([a[M1[i]] for i in range(len(M1))])

    # print("ai =", ai)
    # print("aj =", aj)
    # np,show(0)
    rate = np.zeros((N));
    error = np.zeros((N));
    def sum_rate(x): # x[0]就是公式里面的a，它为AP广播射频能量共给WD收集的时间百分比，就是给WD充电的时间
        for i in range(len(M0)):
            temp_id = M0[i]
            # print("wi[i] =", wi[i])
            # print("wi[i]*eta1*(hi/ki)**(1.0/3)*x[0]**(1.0/3) =", wi[i]*eta1*(hi[i]/ki)**(1.0/3)*x[0]**(1.0/3))
            '''
            print("wi[i] =", wi[i])
            # eta1=((u*p)**(1.0/3))/o # -->本地计算中的n1-->固定参数
            print("u =", u)
            print("p =", p)
            print("o =", o)
            print("eta1 =", eta1)
            print("hi[i] =", hi[i])
            print("ki =", ki)
            print("(hi[i]/ki)**(1.0/3) =", (hi[i]/ki)**(1.0/3))
            print("x[0] =", x[0])
            print("x[0]**(1.0/3) =", x[0]**(1.0/3))
            '''
            rate[temp_id] = (wi[i]*eta1*(hi[i]/ki)**(1.0/3)*x[0]**(1.0/3))/(d_fact)
            # print("Local_A =", Local_A)
            # print("dataAi[i] =", dataAi[i])
            error[temp_id] = (1.0 - (Local_A))*dataAi[i]# *ai[i]
            # print("error[temp_id] =", error[temp_id])
            # np,show()
        sum1=sum(ai*eta1*(hi/ki)**(1.0/3)*x[0]**(1.0/3)) # 本地计算的计算率总和，wi是节点权重，eta1=((u*p)**(1.0/3))/o
        # print("sum1 =", sum1)
        # print("rate =", rate)
        # np,show()
        sum2=0
        
        for i in range(len(M1)): # x[1:]，为分配给第i个wd用于任务卸载的时间比，
            temp_id = M1[i]
            rate[temp_id] = (wj[i]*epsilon*x[i+1]*np.log(1+eta2*hj[i]**2*x[0]/x[i+1]))/(d_fact)
            # print("dataAi[i] =", dataAi[i])
            error[temp_id] = (1- Edge_A)*dataAj[i] # *aj[i]
            # print("error[temp_id] =", error[temp_id])
            # np,show()
            sum2 += aj[i]*epsilon*x[i+1]*np.log(1+eta2*hj[i]**2*x[0]/x[i+1]) # 公式3，计算边缘计算的计算率
        '''
        print(sum(rate))
        print(sum1 + sum2)
        print("rate =", rate)
        print("sum(rate) =", sum(rate))
        print("error =", error)
        print("sum(error) =", sum(error))
        print("w_fac*sum(error) =", w_fac*sum(error))
        print("V*w_fac*sum(error) =", V*w_fac*sum(error))
        print("(sum1+sum2)/d_fact =", (sum1+sum2)/d_fact)
        print("sum(dataA) =", sum(dataA))
        
        np,show()
        '''
        return (sum1+sum2)/d_fact - V*w_fac*sum(error) # 加了个惩罚因子

    def phi(v, j): # 求和公式对tua求导可以得到，tua_j/a # 证明来自：Computation Rate Maximization for Wireless Powered Mobile-Edge Computing with Binary Computation Offloading，附录A
        # return 1/(-1-1/(lambertw(-1/(np.exp(1 + v/wj[j]/epsilon))).real))
        return 1/(-1-1/(lambertw(-1/(np.exp(1 + v/aj[j]/epsilon))).real))

    def p1(v): # AP广播射频能量共wd收集的时间百分比，就是给WD充电的时间
        p1 = 0
        for j in range(len(M1)):
            p1 += hj[j]**2 * phi(v, j)

        return 1/(1 + p1 * eta2)

    def Q(v):  # 求和公式对a的求导，证明来自Computation Rate Maximization for Wireless Powered Mobile-Edge Computing with Binary Computation Offloading，附录B
        # print("wi*eta1*(hi/ki)**(1.0/3) =", ai*wi*eta1*(hi/ki)**(1.0/3))
        # np,show()
        # sum1 = sum(wi*eta1*(hi/ki)**(1.0/3))*p1(v)**(-2/3)/3
        sum1 = sum(ai*eta1*(hi/ki)**(1.0/3))*p1(v)**(-2/3)/3
        sum2 = 0
        for j in range(len(M1)):
            # sum2 += wj[j]*hj[j]**2/(1 + 1/phi(v,j))
            sum2 += aj[j]*hj[j]**2/(1 + 1/phi(v,j))
        return sum1 + sum2*epsilon*eta2 - v

    def tau(v, j): # AP分配给WD用于卸载的时间百分比
        return eta2*hj[j]**2*p1(v)*phi(v,j)

    # bisection starts here
    delta = 0.005
    UB = 999999999
    LB = 0
    while UB - LB > delta:
        v = (float(UB) + LB)/2
        if Q(v) > 0:
            LB = v
        else:
            UB = v

    x.append(p1(v)) # 本地计算时间占比
    for j in range(len(M1)): # 边缘服务器计算时间占比
        x.append(tau(v, j))
    # print("x =", x)
    return sum_rate(x), x[0], x[1:], rate

def Mycd_method(h, Q, weights, V, dataA, Local_A, Edge_A, w_fac, AR):
    N = len(h)
    M_origin = np.random.randint(2,size = N)
    # gain0,a,Tj, _ = bisection(h,M0)
    gain0,a,Tj, _ = Mybisection(h, M_origin, Q, weights, V, dataA, Local_A, Edge_A, w_fac)
    g_list = []
    M_list = []
    g_list.append(gain0)
    M_list.append(M_origin)
    while True:
        for j in range(0,N):
            M = np.copy(M_origin)
            M[j] = (M[j]+1)%2
            # print("M =", M)
            M0=np.where(M==0)[0]  # 取出为0的位置，进行本地计算
            M1=np.where(M==1)[0]  # 取出为1的位置，进行边缘卸载计算
            # print(M0, M1)
            # np,show()
            # gain,a,Tj= bisection(h,M)
            if len(M0)*Local_A + len(M1)*Edge_A >= len(M)*(AR): # 满足精度条件
                # print("meet case")
                gain, a, Tj, _ = Mybisection(h, M, Q, weights, V, dataA, Local_A, Edge_A, w_fac)
                g_list.append(gain)
                M_list.append(M)
        # print("what")
        g_max = max(g_list)
        if g_max > gain0:
            # print("here")
            gain0 = g_max
            M0 = M_list[g_list.index(g_max)]
        else:
            # print("heihei")
            break
    return gain0, M0

# 函数中定义的公式可以在以下论文中查到对应的出处：需要一定数学基础能力
# “Computation rate maximization for wireless powered mobile-edge computing with binary computation ofﬂoading,” 
def bisection(h, M, weights=[]): # 用二分查找实现计算资源分配
    # WD表述无线设备 Wireless Device

    # parameters and equations # 参数和公式的描述
    # 公式1的参数解释
    o=100  # 处理1bit任务数据所需周期数
    p=3  # AP在发射射频能量时的传输功率
    u=0.7 # 能量收集率
    eta1=((u*p)**(1.0/3))/o # -->本地计算中的n1-->固定参数
    ki=10**-26   # 计算能量效率系数
    
    # 公式3的参数解释
    # u为能量收集效率，p为AP在发射射频能量时的传输功率
    eta2=u*p/10**-10 # u*p/N0  -->10**-10为接受噪音功率即N0
    B=2*10**6 # 带宽
    Vu=1.1 # 通信开销
    epsilon=B/(Vu*np.log(2)) # 带宽/通信开销 -->普通对数函数与ln的转换->log2(x) = 1/ln2 * ln(x)
    x = [] # a =x[0]（表示AP给WD充电的时间百分比）, and tau_j = a[1:]（表示分配给需要进行卸载的WD的时间百分比，不需要卸载的WD不考虑在内）
    
    M0=np.where(M==0)[0]  # 取出为0的位置，进行本地计算
    M1=np.where(M==1)[0]  # 取出为1的位置，进行边缘卸载计算
    
    hi=np.array([h[i] for i in M0])  # 信道选择（信道增益是不同的）
    hj=np.array([h[i] for i in M1])  # 信道选择
    

    if len(weights) == 0: # 每个WD的权重是不一样的，可以自行改变
        # default weights [1, 1.5, 1, 1.5, 1, 1.5, ...]
        weights = [1.5 if i%2==1 else 1 for i in range(len(M))]
        
    wi=np.array([weights[M0[i]] for i in range(len(M0))])
    wj=np.array([weights[M1[i]] for i in range(len(M1))])
    
    
    def sum_rate(x): # x[0]就是公式里面的a，它为AP广播射频能量共给WD收集的时间百分比，就是给WD充电的时间
        sum1=sum(wi*eta1*(hi/ki)**(1.0/3)*x[0]**(1.0/3)) # 本地计算的计算率总和，wi是节点权重，eta1=((u*p)**(1.0/3))/o
        sum2=0
        for i in range(len(M1)): # x[1:]，为分配给第i个wd用于任务卸载的时间比，
            sum2+=wj[i]*epsilon*x[i+1]*np.log(1+eta2*hj[i]**2*x[0]/x[i+1]) # 公式3，计算边缘计算的计算率
        return sum1+sum2
    
    # 公式（17），推导在附录A
    def phi(v, j): # 求和公式对tua求导可以得到，tua_j/a # 证明来自：Computation Rate Maximization for Wireless Powered Mobile-Edge Computing with Binary Computation Offloading，附录A
        return 1/(-1-1/(lambertw(-1/(np.exp(1 + v/wj[j]/epsilon))).real))

    # 公式（18）
    def p1(v): # AP广播射频能量共wd收集的时间百分比，就是给WD充电的时间
        p1 = 0
        for j in range(len(M1)):
            p1 += hj[j]**2 * phi(v, j)

        return 1/(1 + p1 * eta2)

    # 公式（19），推导在附录B
    def Q(v):  # 求和公式对a的求导，证明来自Computation Rate Maximization for Wireless Powered Mobile-Edge Computing with Binary Computation Offloading，附录B
        sum1 = sum(wi*eta1*(hi/ki)**(1.0/3))*p1(v)**(-2/3)/3
        sum2 = 0
        for j in range(len(M1)):
            sum2 += wj[j]*hj[j]**2/(1 + 1/phi(v,j))
        return sum1 + sum2*epsilon*eta2 - v

    # 公式（16）
    def tau(v, j): # AP分配给WD用于卸载的时间百分比
        return eta2*hj[j]**2*p1(v)*phi(v,j)

    # bisection starts here
    # 算法1
    delta = 0.005
    UB = 999999999
    LB = 0
    while UB - LB > delta:
        v = (float(UB) + LB)/2
        if Q(v) > 0:
            LB = v
        else:
            UB = v

    x.append(p1(v)) # 充电时间占比
    for j in range(len(M1)): # WD卸载时间占比
        x.append(tau(v, j))
        
    return sum_rate(x), x[0], x[1:] # 加权计算率，充电时间占比，WD卸载时间占比


# cd_method方案来自论文“Computation rate maximization for wireless powered mobile-edge computing with binary computation ofﬂoading,” 
def cd_method(h):
    N = len(h)
    M0 = np.random.randint(2,size = N) # 随机生成卸载决策
    gain0,a,Tj= bisection(h,M0) # 得到当前卸载决策的增益
    g_list = []
    M_list = []
    while True:
        for j in range(0,N):
            M = np.copy(M0)
            M[j] = (M[j]+1)%2 # 更新卸载决策
            gain,a,Tj= bisection(h,M) # 得到更新卸载决策的增益
            g_list.append(gain) # 增益列表
            M_list.append(M) # 卸载决策列表
        g_max = max(g_list) # 取最大增益
        if g_max > gain0: 
            gain0 = g_max # 更新最大增益
            M0 = M_list[g_list.index(g_max)] # 更新最大增益对应的卸载决策
        else:
            break
    return gain0, M0 # 最大增益及其最佳卸载决策