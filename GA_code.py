import numpy as np
import random
# 交叉(交配)过程
def crossover(popDNA_m, popDNA_copy, dNA_SIZE, pOP_SIZE, CROSS_RATE):
    # print "parent =\n", parent
    # print "pop =\n", pop
    # 交配概率
    # print("popDNA_m =", popDNA_m)
    if np.random.rand() < CROSS_RATE:
        # select another individual from pop
        # 从群体中选择另一个个体
        # size 表示生成几个数
        i_ = np.random.randint(0, pOP_SIZE, size=1)
        # print("i_ =", i_)
        # choose crossover points
        # 选择交叉的节点,以True or False 形式存在
        # size 表示生成几个数
        # print "dNA_SIZE =", dNA_SIZE
        cross_points = np.random.randint(0, 2, size=dNA_SIZE).astype(np.bool)
        # print("cross_points =", cross_points)
        # cross_points[1] = True
        # mating and produce one child
        # 生成孩子，作为下一代的父母
        # 将pop[i_, cross_points]赋值给parent[cross_points]
        popDNA_m[cross_points] = popDNA_copy[i_, cross_points]
        # print("popDNA_copy[i_, cross_points] =", popDNA_copy[i_, cross_points])
        # print("popDNA_m[cross_points] =", popDNA_m[cross_points])
        '''
        delete = popDNA_m[2: dNA_SIZE]
        delete_temp = []
        for i in range(len(delete)):
            delete_temp.append(delete[i])
        # 能够保证每个接货点都还存在
        x = random.sample(RemainNodeList, dNA_SIZE - 2)
        for i in range(len(x)):
            if x[i] not in delete_temp:
                for j in range(len(delete_temp)):
                    if delete_temp.count(delete_temp[j]) >= 2:
                        delete_temp[delete_temp.index(delete_temp[j])] = x[i]
        popDNA_m[2: dNA_SIZE] = delete_temp
        '''
    # 生成孩子
    # child = parent
    # print("popDNA_m =", popDNA_m)
    # np,show()
    return popDNA_m
# 变异
# 变异过程
def mutate(childDNA, dNA_SIZE, MUTATION_RATE):
    # DNA中任意一个点
    # childDNA = [1, 0, 6, 9, 3, 8 ,4, 7,2,5, 1]
    # print("childDNA =", childDNA)
    for point in range(dNA_SIZE):
        # 从DNA中突变某一节点，MUTATION_RATE突变概率
        # 0 变 1 ， 1变 0
        # 确保所突变的点不能为仓库,仓库位于基因的第1个位置
        if (np.random.rand() < MUTATION_RATE): # and (point != 1):
            # 所选用的货车，在第0位，用0表示载重量为2t的货车
            # if point == 0:#过于简单，很难产生有效的解
            childDNA[point] = 1 if childDNA[point] == 0 else 0
            # 其余的point点为接货点
            # else:
            # 随机抽样接货点,返回闭区间的值
            # childDNA[point] = random.randint(1, dNA_SIZE - 2)
            # childDNA_temp = random.sample(RemainNodeList, 1)
            # print("childDNA_temp =", childDNA_temp)
            # childDNA[point] = childDNA_temp[0]

    '''
    childDNAList = childDNA[2:]
    childDNAList_temp = []
    for c_i in range(len(childDNAList)):
        childDNAList_temp.append(childDNAList[c_i])
    # 能够保证每个接货点都还存在
    x = random.sample(RemainNodeList, dNA_SIZE - 2)
    # 保证接货点的完整
    for i in range(len(x)):
        if x[i] not in childDNAList_temp:
            for j in range(len(childDNAList_temp)):
                if childDNAList_temp.count(childDNAList_temp[j]) >= 2:
                    childDNAList_temp[childDNAList_temp.index(childDNAList_temp[j])] = x[i]
    childDNA[2: dNA_SIZE] = childDNAList_temp
    # 孩子成长
    '''
    # print("childDNA =", childDNA)
    # np,show()
    return childDNA
def GA(popDNA):
    CROSS_RATE = 0.8  # mating probability (DNA crossover)
    MUTATION_RATE =  0.003  # mutation probability
    dNA_SIZE = len(popDNA[0][0:])
    pOP_SIZE = len(popDNA)
    # print(popDNA)
    # print(dNA_SIZE)
    # print(pOP_SIZE)
    # np,show()
    # print("M_DNA =", popDNA, type(popDNA))


    # 对x, y进行自然选择
    # popDNA = select(popDNA, fitness, pOP_SIZE)

    # 复制群体
    popDNA_copy = popDNA.copy()
    for m in range(0, len(popDNA)):
        # 交叉
        childx = crossover(popDNA[m], popDNA_copy, dNA_SIZE, pOP_SIZE, CROSS_RATE)
        # 突变
        childx = mutate(childx, dNA_SIZE, MUTATION_RATE)
        # parent is replaced by its child
        # 孩子代替父母
        # DNA的每一位都在被替换了
        popDNA[m][:] = childx
    return popDNA
    # print("popDNA =", popDNA)

if __name__ == "__main__":
    N = 10 # 用户个数
    K = 20 #种群数量
    M = np.zeros(N)
    
    for i in range(N):
        if i%2 == 0:
            print()
            M[i] = 1
        else:
            M[i] = 0

    popDNA = GA(M, K)
    print("popDNA =", popDNA)
    '''
    M_temp = np.zeros(N)
    for j in range(len(M)):
        M_temp[j] = 1 if M[j] == 0 else 0
    print("M =", M, type(M))
    print("M_temp =", M_temp, type(M_temp))
    dNA_SIZE = len(M)
    popDNA = np.zeros((K, N))
    
    CROSS_RATE = 0.8  # mating probability (DNA crossover)
    MUTATION_RATE =  0.003  # mutation probability
    for i in range(K):
        if i%2 == 0:
            popDNA[i][0:] = M
        else:
            popDNA[i][0:] = M_temp
    pOP_SIZE = len(popDNA)
    NodeList = []
    print(dNA_SIZE)
    print(pOP_SIZE)
    print("M_DNA =", popDNA, type(popDNA))


    # 对x, y进行自然选择
    # popDNA = select(popDNA, fitness, pOP_SIZE)

    # 复制群体
    popDNA_copy = popDNA.copy()
    # print("popDNA[0] =", popDNA[0])
    # print("popDNA =", popDNA)
    # print("dNA_SIZE =", dNA_SIZE)
    # print("pOP_SIZE =", pOP_SIZE)
    # print("NodeList =", NodeList)
    # 从全体中选父母用于产生后代群体
    for m in range(0, len(popDNA)):
        # 交叉
        childx = crossover(popDNA[m], popDNA_copy, dNA_SIZE, pOP_SIZE, NodeList)
        # 突变
        childx = mutate(childx, dNA_SIZE, NodeList)
        # parent is replaced by its child
        # 孩子代替父母
        # DNA的每一位都在被替换了
        popDNA[m][:] = childx
    print("popDNA =", popDNA)
    '''