 

import numpy as np
import itertools

class individual:
    def __init__(self,genes):
        self.genes = genes


    def cross_over(self,ind):
        k = np.random.randint(0,self.genes.__len__())
        new_genes = self.genes[:k] + ind.genes[k:]
        return individual(new_genes)

    def mutate(self,mutation_rate):
        new_genes = []
        for g in self.genes:
            if(np.random.rand()<mutation_rate):
                new_genes.append(np.random.randint(0,10))
            else:
                new_genes.append(g)

        return individual(new_genes)

    def get_ans(self):
        return self.genes





def rand_individual(len):
    genes=[]
    for i in range(0,len):
        genes.append(np.random.randint(0,10))
    return individual(genes)






class population:
    def __init__(self,n,n_genes,fitness_function):
        self.n=n
        self.n_genes=n_genes
        self.fitness_function = fitness_function
        self.individuals=[]

        for i in range(0,n):
            self.individuals.append(rand_individual(n_genes))

    def evaluate_fitness(self,solution):
        pfit = []
        for i in self.individuals:
            ians = i.get_ans()
            ifit= self.fitness_function(ians,solution)
            pfit.append(ifit)
        return pfit

    def selection(self,solution,fperc):
        mating_pool =[]
        pop_fitness = self.evaluate_fitness(solution)
        n_parent = int(1 - self.n * fperc)

        fittest_index = np.argsort(pop_fitness)[n_parent:][::-1]
        #print(pop_fitness)
        #print(fittest_index)

        for index in fittest_index:
            mating_pool.append(self.individuals[index])
            #print(self.individuals[index].get_ans())

        print(self.individuals[0].get_ans())
        return mating_pool




    def cross_over(self,mating_pool):
        children = []

        for comb in list(itertools.combinations(mating_pool, 2)):
            children.append(comb[0].cross_over(comb[1]))

        while(children.__len__() < self.n):
            rand_parent0 = mating_pool[np.random.randint(0,mating_pool.__len__())]
            rand_parent1 = mating_pool[np.random.randint(0,mating_pool.__len__())]
            children.append(rand_parent0.cross_over(rand_parent1))

        return children[:self.n]


    def mutation(self,children,mutation_rate):
        offspring = []
        for c in children:
            offspring.append(c.mutate(mutation_rate))
        return offspring




def fitness_function(answer,solution):
    fit = 0
    for a,s in zip(answer,solution):
        if(a==s):
            fit+=1
    return fit




def evolve():
    solution = [0, 1, 2, 3, 4]
    n_genes = 5

    pop_size = 10

    fperc = 0.5
    mutation_rate = 0.1

    pop =population(pop_size,n_genes,fitness_function)

    iterations =0
    while(iterations<100):

        mating_pool = pop.selection(solution,fperc)

        children = pop.cross_over(mating_pool)
        pop.individuals = pop.mutation(children,mutation_rate)

        iterations+=1



evolve()


