"""
    This is a NP optimization problem similar to bin packing, knapsack, etc...
    Requires either linear programming or ML such as genetic algorithm, random hill climbing, simulated annealing, or MIMIC

    restraints:
        1 of each pizza type AT MOST
        No more than max_slices of pizza

    max_slices = max slices of pizza that can be ordered (a restraint on y)
    pizza_types = number of pizza types (n)
    slice_counts = the count of slices for each pizza type (x)

    The equation to optimize:
        y = w1x1 + ... + wnxn
        y <= max_slices
        w = 1 | 0
    
    each pizza type will have a weight of 1 or 0 representing if that pizza is included in the order, if y <= max_slices then we have a viable solution.

    The goal is to optimize the weights (w) of this equation.

    references:
        https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6
        https://www.youtube.com/watch?v=uCXm6avugCo
        https://www.geeksforgeeks.org/genetic-algorithm-for-reinforcement-learning-python-implementation/


    proposed changes:
        - replace single crossover with random crossover for the larger datasets
        - add random mutations
        - add epochs for the larger datasets
        - increase mutation rate for the larger datasets
        - increase amount of mutations for the larger datasets


"""
import numpy as np
import random


def gen_population(pizza_types,pop_size=100):
    options = np.array([i for i in range(pizza_types)])

    pop = np.random.randint(2,size=pizza_types)
    for i in range(pop_size-1):
        elem = np.random.randint(2,size=pizza_types)
        pop = np.vstack((pop,elem))

    return pop
     
def fitness(pop,slice_counts,max_slices):
    """
        More pizza slices means a higher fitness.
        silces over max_slices get a fitness of -1.
        this is maximization so higher is better.
    """
    fit = np.sum(pop*slice_counts,axis=1)
    fit = np.where(fit <= max_slices, fit, -1)
    return fit

def crossover_single(parents,pop_size):
    """
        simple single point cross over, half of genes from one parent and half from the other.
        each parent is randomly selected without replacement
    """
    offspring = np.empty((0,parents.shape[1]))
    crossover_point = int(parents.shape[1]/2)
    for i in range(pop_size):
        parent_1,parent_2 = np.random.choice(parents.shape[0],2,replace=False)
        child = np.append(parents[parent_1,:crossover_point],parents[parent_2,crossover_point:])
        offspring = np.vstack((offspring,child))
    return offspring

def crossover_random(parents,pop_size):
    """
        random genes selected from each parent equally
    """
    offspring = np.empty((0,parents.shape[1]))
    crossover_point = int(parents.shape[1]/2)
    for i in range(pop_size):
        parent_1,parent_2 = np.random.choice(parents.shape[0],2,replace=False)
        choice = np.random.randint(2,size=parents.shape[1]).astype(bool)
        child = np.where(choice,parents[parent_1],parents[parent_2])
        offspring = np.vstack((offspring,child))
    return offspring

def mutate_single(pop,mutation_chance=0.5):
    """
        has mutation chance to mutate a single gene
    """
    for i in range(pop.shape[0]):
        if random.random() < mutation_chance:
            pop[i,np.random.choice(pop.shape[1])] = np.random.randint(2)
    return pop

def mutate_random(pop,mutation_chance=0.5):
    """
        has mutation chance to mutate 1/2 of the genes randomly
    """
    for i in range(pop.shape[0]):
        if random.random() < mutation_chance:
            for j in np.random.choice(pop.shape[1],int(pop.shape[1]/2),replace=False):
                pop[i,j] = np.random.randint(2)
    return pop

def mate(pop,fit,selection=0.25):
    """
        the top 40% of the population will be mated and create children to fill in the missing 60%
    """
    top_count = int(fit.shape[0]*selection)
    top = np.argsort(fit)[::-1]
    top = top[:top_count]
    best = pop[top]

    offspring = crossover_single(best,fit.shape[0]-top_count)
    # offspring = crossover_random(best,fit.shape[0]-top_count)
    # print('offspring',offspring)

    mutated = mutate_single(offspring,mutation_chance=1.0)
    # mutated = mutate_random(offspring,mutation_chance=1.0)
    # print('mutated:',mutated)
    
    new_pop = np.vstack((best,mutated))
    return new_pop

def select_best(pop,slice_counts,max_slices):
    fit = fitness(pop,slice_counts,max_slices)
    top = np.argsort(fit)[::-1]
    best = pop[top[0]]
    return best,fit[top[0]]


def run(filename):
    # read in data
    with open(filename,'r') as f:
        data = f.read()
    items = data.split('\n')
    max_slices, pizza_types = items[0].split(' ')
    slice_counts = items[1]
    # format input
    max_slices = int(max_slices)
    pizza_types = int(pizza_types)
    slice_counts = [int(i) for i in slice_counts.split(' ')]
    
    # generate initial population
    pop = gen_population(pizza_types)
    epochs = 200

    # run genetic algorithm
    for i in range(epochs):
        fit = fitness(pop,slice_counts,max_slices)
        new_pop = mate(pop,fit)
        best,best_fitness = select_best(pop,slice_counts,max_slices)
        print('epoch: {} , best fitness {}'.format(i,best_fitness))

    print('max slices:',max_slices)
    print('pizza types:',pizza_types)
    # print('slice counts:',slice_counts)
    print('score: {}'.format(best_fitness))
    # print('solution: {}'.format(best[:10]))

    # format output
    to_order = str(np.sum(best))
    selections = np.nonzero(best)[0].tolist()
    selections = ' '.join([str(i) for i in selections])

    # print('final solution:\n\t{}\n\t{}'.format(to_order,selections))

    # write out solution
    with open(filename[:-3]+'.out','w') as f:
        f.write(to_order)
        f.write('\n')
        f.write(selections)
        f.write('\n')



if __name__ == '__main__':
    """
        uncomment the file you wish to run
    """
    # filename = 'a_example.in'
    filename = 'b_small.in'
    # filename = 'c_medium.in'
    # filename = 'd_quite_big.in'
    # filename = 'e_also_big.in'
    run(filename)