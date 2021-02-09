import numpy as np
import matplotlib.pyplot as plt
import time
import os

def individual_distance(ind):
    return np.sum(dists[ind,ind.take(np.arange(1,M+1),mode="wrap")])
def distances(pop):
    cur_distances = np.zeros(pop.shape[0])
    for i, p in enumerate(pop):
        cur_distances[i] = individual_distance(p)
    return cur_distances

# Binary tournament selection
def bts_gg(pop):
    newpop = np.zeros((N, M)).astype(np.int16)
    lens = distances(pop)
    for i in range(N):
        p_a, p_b = np.random.choice(N, size=2, replace=False)
        p_1 = p_a if lens[p_a] < lens[p_b] else p_b
        newpop[i] = pop[p_1]
    return newpop

# Crossover
def crossover(ind_a, ind_b):
    """
    Applies crossover to two individuals
    """
    # Create empty array
    offspring = np.zeros((2, M)).astype(np.int16)

    # Select cutpoints
    cuts = np.random.choice(M+1, size=2, replace=False)
    cutlen = np.abs(cuts[0]-cuts[1])

    # Take slices of parents and assign them to offspring
    slice_a = ind_a[min(cuts):max(cuts)]
    slice_b = ind_b[min(cuts):max(cuts)]
    offspring[0,:cutlen] = slice_a
    offspring[1,:cutlen] = slice_b

    a_needs = np.setdiff1d(ind_b, slice_a, assume_unique=True)
    b_needs = np.setdiff1d(ind_a, slice_b, assume_unique=True)
    offspring[0,cutlen:] = a_needs
    offspring[1,cutlen:] = b_needs

    thisrange = np.arange(start = M-min(cuts),stop=2*M-min(cuts))
    offspring[0] = np.take(offspring[0],thisrange,mode="wrap")
    offspring[1] = np.take(offspring[1],thisrange,mode="wrap")
    return offspring

def crossover_pop(pop):
    """
    Applies crossover to every individual in a population
    """
    if p_c == 0:
        return pop
    newpop = np.zeros((N, M)).astype(np.int16)
    for i in range(0, N, 2):
        ind_a, ind_b = np.random.choice(N, size=2, replace=False)
        if np.random.rand(1) < p_c:
            newpop[i:i+2] = crossover(pop[ind_a], pop[ind_b])
        else:
            newpop[i] = pop[ind_a]
            newpop[i+1] = pop[ind_b]
    return newpop

# Mutate
def swap_ind(ind, a, b):
    temp = ind[a]
    ind[a] = ind[b]
    ind[b] = temp
def swap(pop, i, a, b):
    temp = pop[i, a]
    pop[i, a] = pop[i, b]
    pop[i, b] = temp
def mutate_pop(pop):
    for i in range(N):
        if np.random.rand(1) < p_m:
            i_a, i_b = np.random.choice(M, size=2, replace=False)
            swap(pop, i, i_a, i_b)
    return pop

# Local search
def flip(ind, i, j):
    ind[i:j] = np.flip(ind[i:j])
    return ind
def local_search(ind, quick_break = False):
    best_dist = individual_distance(ind)
    best_flip = (0, 0)
    for i in range(M):
        for j in range(i, M):
            ind = flip(ind, i, j)
            if individual_distance(ind) < best_dist:
                if quick_break:
                    return ind
                else:
                    best_flip = (i, j)
                    best_dist = individual_distance(ind)
            ind = flip(ind, i, j)
    ind = flip(ind, *best_flip)
    return ind
def local_search_pop(pop, quick_break = False):
    for i in range(N):
        pop[i] = local_search(pop[i], quick_break)
    return pop

N = 20
p_c = 0.7
p_m = 0.1
def run_GA(data, do_local_search = False):
    # Create distance matrix
    x_diff = np.subtract.outer(data[:, 0], data[:, 0])
    y_diff = np.subtract.outer(data[:, 1], data[:, 1])
    dists_sq = np.add(np.square(x_diff), np.square(y_diff))
    global dists
    dists = np.sqrt(dists_sq)

    # Random initialization
    global M
    M = data.shape[0]

    global pop
    pop = np.zeros((N, M)).astype(np.int16)
    for i in range(N):
        pop[i] = np.random.permutation(M)

    # Without local search
    iters = 250
    fitnesses = np.zeros((2, iters))
    for i in range(iters):
        pop = mutate_pop(crossover_pop(bts_gg(pop)))
        if do_local_search:
            pop = local_search_pop(pop, quick_break = False)
        fitness = distances(pop)
        fitnesses[0, i] = np.min(fitness)
        fitnesses[1, i] = np.mean(fitness)
    best = pop[np.argmin(fitness)]

    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(fitnesses[0], label="best")
    ax.plot(fitnesses[1], label="mean")
    ax.set_xlabel("Iteration")
    ax.set_xlabel("Distance")
    plt.legend()
    plt.savefig(f"{os.getcwd()}/fitness_{M}_LS={do_local_search}.png")

    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(1,1,1)
    bestroute = data[best]
    ax.plot(bestroute[:,0], bestroute[:,1])
    plt.savefig(f"{os.getcwd()}/route_{M}_LS={do_local_search}.png")



data = np.loadtxt( "Data/file-tsp.txt")
run_GA(data, False)
run_GA(data, True)

data = np.loadtxt( "Data/27_cities.txt")
run_GA(data, False)
run_GA(data, True)