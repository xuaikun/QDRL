# -*- coding: utf-8 -*-
import numpy as np
from scipy import optimize
from scipy.special import lambertw
import scipy.io as sio                     
import time


def plot_gain( gain_his): 
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

    # parameters and equations
    o=100  #
    # p=3  # 
    p = 3 # W
    u = 0.7 # 
    eta1=((u*p)**(1.0/3))/o # 

    d_fact = 10**6
    ki=(10**-26)  
    
    ch_fact = 10**10
    B=2*10**6 #
    eta2=u*p/(10**(-10)) # u*p/N0  -->10**-10
    Vu=1.1 # 
    epsilon=B/(Vu*np.log(2)) #  -->log2(x) = 1/ln2 * ln(x)
    x = [] # a =x[0], and tau_j = a[1:]
    
    M0=np.where(M==0)[0] 
    M1=np.where(M==1)[0] 

    hi=np.array([h[i] for i in M0])  # 
    hj=np.array([h[i] for i in M1])  # 
    dataAi=np.array([dataA[i] for i in M0])  #
    dataAj=np.array([dataA[i] for i in M1])  
    if len(weights) == 0:
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

    rate = np.zeros((N));
    error = np.zeros((N));
    def sum_rate(x): 
        for i in range(len(M0)):
            temp_id = M0[i]
            rate[temp_id] = (wi[i]*eta1*(hi[i]/ki)**(1.0/3)*x[0]**(1.0/3))/(d_fact)
            error[temp_id] = (1.0 - (Local_A))*dataAi[i]# *ai[i]
        sum1=sum(ai*eta1*(hi/ki)**(1.0/3)*x[0]**(1.0/3)) #
        sum2=0
        
        for i in range(len(M1)): 
            temp_id = M1[i]
            rate[temp_id] = (wj[i]*epsilon*x[i+1]*np.log(1+eta2*hj[i]**2*x[0]/x[i+1]))/(d_fact)
            error[temp_id] = (1- Edge_A)*dataAj[i] # *aj[i]
            sum2 += aj[i]*epsilon*x[i+1]*np.log(1+eta2*hj[i]**2*x[0]/x[i+1]) # 
        return (sum1+sum2)/d_fact - V*w_fac*sum(error) #
    return sum_rate(alpha_tau), alpha_tau[0], alpha_tau[1:], rate

def alpha_Mycd_method(h, Q, weights, V, dataA, Local_A, Edge_A, w_fac, alpha_tau, AR):
    N = len(h) 
    M_origin = np.random.randint(2,size = N)
    M0=np.where(M_origin==0)[0]  #
    M1=np.where(M_origin==1)[0]  # 
    while len(M0)*Local_A + len(M1)*Edge_A < len(M_origin)*(AR): # 
        M_origin = np.random.randint(2,size = N)
        M0=np.where(M_origin==0)[0]  # 
        M1=np.where(M_origin==1)[0]  # 
    
    gain0,a,Tj, _ = Mybisection(h, M_origin, Q, weights, V, dataA, Local_A, Edge_A, w_fac)
   
    g_list = []
    M_list = []
    g_list.append(gain0)
    M_list.append(M_origin)
   
    while True:
        for j in range(0,N):
            M = np.copy(M_origin)
            M[j] = (M[j]+1)%2
            
            M0=np.where(M==0)[0]  
            M1=np.where(M==1)[0] 
            if len(M0)*Local_A + len(M1)*Edge_A >= len(M)*(AR): 
                gain, a, Tj, _ = Mybisection(h, M, Q, weights, V, dataA, Local_A, Edge_A, w_fac)
                g_list.append(gain)
                M_list.append(M)
        g_max = max(g_list)
        if g_max > gain0:
            gain0 = g_max
            M_origin = M_list[g_list.index(g_max)]
        else:
            break
    M = M_origin
    gain0, a, Tj, _ = Mybisection(h, M, Q, weights, V, dataA, Local_A, Edge_A, w_fac)
    return gain0, a, Tj, _, M

def Mybisection(h, M, Q, weights, V, dataA, Local_A, Edge_A, w_fac):
    # parameters and equations
    o=100  #
    p = 3 # W
    u = 0.7 # 
    eta1=((u*p)**(1.0/3))/o #
    d_fact = 10**6
    ki=(10**-26)   # 
    
    ch_fact = 10**10
    
    B=2*10**6 # 
    eta2=u*p/(10**(-10)) # u*p/N0  -->10**-10
   
    Vu=1.1 # 
    epsilon=B/(Vu*np.log(2)) # -->log2(x) = 1/ln2 * ln(x)
    x = [] # a =x[0], and tau_j = a[1:]
    
    M0=np.where(M==0)[0]  #
    M1=np.where(M==1)[0]  #
    
    hi=np.array([h[i] for i in M0])  # 
    hj=np.array([h[i] for i in M1])  # 

    dataAi=np.array([dataA[i] for i in M0])  # 
    dataAj=np.array([dataA[i] for i in M1])  # 

    if len(weights) == 0:
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

    rate = np.zeros((N));
    error = np.zeros((N));
    def sum_rate(x): 
        for i in range(len(M0)):
            temp_id = M0[i]
            rate[temp_id] = (wi[i]*eta1*(hi[i]/ki)**(1.0/3)*x[0]**(1.0/3))/(d_fact)
            error[temp_id] = (1.0 - (Local_A))*dataAi[i]# *ai[i]
        sum1=sum(ai*eta1*(hi/ki)**(1.0/3)*x[0]**(1.0/3)) # eta1=((u*p)**(1.0/3))/o
        sum2=0
        
        for i in range(len(M1)): # x[1:]，
            temp_id = M1[i]
            rate[temp_id] = (wj[i]*epsilon*x[i+1]*np.log(1+eta2*hj[i]**2*x[0]/x[i+1]))/(d_fact)
            
            error[temp_id] = (1- Edge_A)*dataAj[i] # *aj[i]
            
            sum2 += aj[i]*epsilon*x[i+1]*np.log(1+eta2*hj[i]**2*x[0]/x[i+1]) #
        return (sum1+sum2)/d_fact - V*w_fac*sum(error) #

    def phi(v, j): # 
        
        return 1/(-1-1/(lambertw(-1/(np.exp(1 + v/aj[j]/epsilon))).real))

    def p1(v): # 
        p1 = 0
        for j in range(len(M1)):
            p1 += hj[j]**2 * phi(v, j)

        return 1/(1 + p1 * eta2)

    def Q(v):  # ，
        sum1 = sum(ai*eta1*(hi/ki)**(1.0/3))*p1(v)**(-2/3)/3
        sum2 = 0
        for j in range(len(M1)):
            sum2 += aj[j]*hj[j]**2/(1 + 1/phi(v,j))
        return sum1 + sum2*epsilon*eta2 - v

    def tau(v, j):
        return eta2*hj[j]**2*p1(v)*phi(v,j)
    delta = 0.005
    UB = 999999999
    LB = 0
    while UB - LB > delta:
        v = (float(UB) + LB)/2
        if Q(v) > 0:
            LB = v
        else:
            UB = v

    x.append(p1(v)) 
    for j in range(len(M1)): 
        x.append(tau(v, j))
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
            M0=np.where(M==0)[0]  
            M1=np.where(M==1)[0]  
            if len(M0)*Local_A + len(M1)*Edge_A >= len(M)*(AR): # 
                gain, a, Tj, _ = Mybisection(h, M, Q, weights, V, dataA, Local_A, Edge_A, w_fac)
                g_list.append(gain)
                M_list.append(M)
        g_max = max(g_list)
        if g_max > gain0:
            gain0 = g_max
            M0 = M_list[g_list.index(g_max)]
        else:
            break
    return gain0, M0

def bisection(h, M, weights=[]): #
    # parameters and equations # 
    o=100  # 
    p=3  # 
    u=0.7 # 
    eta1=((u*p)**(1.0/3))/o # --
    ki=10**-26   # 
    
    eta2=u*p/10**-10 # u*p/N0  -->10**-10
    B=2*10**6 # 
    Vu=1.1 # 
    epsilon=B/(Vu*np.log(2)) # log2(x) = 1/ln2 * ln(x)
    x = [] # a =x[0], and tau_j = a[1:]
    
    M0=np.where(M==0)[0]  # 
    M1=np.where(M==1)[0]  # 
    
    hi=np.array([h[i] for i in M0])  # 
    hj=np.array([h[i] for i in M1])  # 
    

    if len(weights) == 0: #
        weights = [1.5 if i%2==1 else 1 for i in range(len(M))]
        
    wi=np.array([weights[M0[i]] for i in range(len(M0))])
    wj=np.array([weights[M1[i]] for i in range(len(M1))])
    
    
    def sum_rate(x): #
        sum1=sum(wi*eta1*(hi/ki)**(1.0/3)*x[0]**(1.0/3)) # eta1=((u*p)**(1.0/3))/o
        sum2=0
        for i in range(len(M1)): # x[1:]，，
            sum2+=wj[i]*epsilon*x[i+1]*np.log(1+eta2*hj[i]**2*x[0]/x[i+1]) # 
        return sum1+sum2

    def phi(v, j): # ，tua_j/a # 
        return 1/(-1-1/(lambertw(-1/(np.exp(1 + v/wj[j]/epsilon))).real))

    def p1(v): 
        p1 = 0
        for j in range(len(M1)):
            p1 += hj[j]**2 * phi(v, j)

        return 1/(1 + p1 * eta2)

    def Q(v):  
        sum1 = sum(wi*eta1*(hi/ki)**(1.0/3))*p1(v)**(-2/3)/3
        sum2 = 0
        for j in range(len(M1)):
            sum2 += wj[j]*hj[j]**2/(1 + 1/phi(v,j))
        return sum1 + sum2*epsilon*eta2 - v

    def tau(v, j): #
        return eta2*hj[j]**2*p1(v)*phi(v,j)

    delta = 0.005
    UB = 999999999
    LB = 0
    while UB - LB > delta:
        v = (float(UB) + LB)/2
        if Q(v) > 0:
            LB = v
        else:
            UB = v

    x.append(p1(v)) # 
    for j in range(len(M1)): #
        x.append(tau(v, j))
        
    return sum_rate(x), x[0], x[1:] #

def cd_method(h):
    N = len(h)
    M0 = np.random.randint(2,size = N) 
    gain0,a,Tj= bisection(h,M0) #
    g_list = []
    M_list = []
    while True:
        for j in range(0,N):
            M = np.copy(M0)
            M[j] = (M[j]+1)%2 
            gain,a,Tj= bisection(h,M) # 
            g_list.append(gain) # 
            M_list.append(M) # 
        g_max = max(g_list) # 
        if g_max > gain0: 
            gain0 = g_max #
            M0 = M_list[g_list.index(g_max)]
        else:
            break
    return gain0, M0 
