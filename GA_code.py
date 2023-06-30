import numpy as np
import random
# crossover
def crossover(popDNA_m, popDNA_copy, dNA_SIZE, pOP_SIZE, CROSS_RATE):
    if np.random.rand() < CROSS_RATE:
        # select another individual from pop
        i_ = np.random.randint(0, pOP_SIZE, size=1)
        # choose crossover points
        cross_points = np.random.randint(0, 2, size=dNA_SIZE).astype(np.bool)
        # mating and produce one child
        popDNA_m[cross_points] = popDNA_copy[i_, cross_points]
    return popDNA_m
# mutation
def mutate(childDNA, dNA_SIZE, MUTATION_RATE):
    for point in range(dNA_SIZE):
        if (np.random.rand() < MUTATION_RATE): # and (point != 1):
            childDNA[point] = 1 if childDNA[point] == 0 else 0
    return childDNA
def GA(popDNA):
    CROSS_RATE = 0.8  # mating probability (DNA crossover)
    MUTATION_RATE =  0.003  # mutation probability
    dNA_SIZE = len(popDNA[0][0:])
    pOP_SIZE = len(popDNA)
    popDNA_copy = popDNA.copy()
    for m in range(0, len(popDNA)):
        childx = crossover(popDNA[m], popDNA_copy, dNA_SIZE, pOP_SIZE, CROSS_RATE)
        childx = mutate(childx, dNA_SIZE, MUTATION_RATE)
        popDNA[m][:] = childx
    return popDNA

if __name__ == "__main__":
    N = 10 
    K = 20 
    M = np.zeros(N)
    for i in range(N):
        if i%2 == 0:
            print()
            M[i] = 1
        else:
            M[i] = 0

    popDNA = GA(M, K)
    print("popDNA =", popDNA)
