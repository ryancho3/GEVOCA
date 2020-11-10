from Agent import Agent
import copy as copy
import random

class Population:

    def __init__(self, k, t, r, p, size, mutationrate):
        self.K = k
        self.T = t
        self.r = r
        self.p = p
        self.s = size
        self.m = mutationrate

    def initialize(self):
        self.agents = []
        for i in range(self.s):
            self.agents.append(Agent(self.K, self.T, self.r, self.p))
            self.agents[i].initialize()


    def calcallfitness(self):
        for i in range(len(self.agents)):
            self.agents[i].calcfitness()

    def adjustall(self):
        for i in range(len(self.agents)):
            self.agents[i].adjust()
    def setpopulation(self, pop):
        self.agents = pop

    def calctotalfitness(self):
        tot = 0.0
        for i in range(len(self.agents)):
            tot = tot + self.agents[i].fitness
        return tot

    def getbestagent(self):

        max = 0
        index = 0
        for i in range (len(self.agents)):
            if (self.agents[i].fitness > max):
                index = i
                max = self.agents[i].fitness
        return self.agents[index]


    def getaveragefitness(self):
        return self.calctotalfitness()/len(self.agents)

    def selection(self):
        tot = self.calctotalfitness()
        threshold = random.random()*tot
        add = 0.0

        for i in range(self.s):
            add += self.agents[i].fitness
            if (add>threshold):
                p1 = self.agents[i]

        add = 0.0
        threshold = random.random() * tot
        for j in range(self.s):
            add += self.agents[j].fitness
            if (add>threshold):
                p2 = self.agents[j]

        return [p1, p2]

    def generateoffspring(self):
        parents = self.selection()
        genome = self.crossover(parents[0].getgenome(), parents[1].getgenome())
        offspring = Agent(self.K, self.T, self.r, self.p)
        offspring.setgenome(genome)
        return offspring

    def crossover(self, g1, g2):
        genelength = len(g1)
        crosspoint = random.randint(0, genelength)
        genome = [0]*genelength

        for i in range(crosspoint):
            genome[i] = g1[i]

        for j in range(crosspoint, genelength):
            genome[j] = g2[j]

        return genome

    def nextgeneration(self):
        pop = []
        pop.append(copy.deepcopy(self.getbestagent()))
        for i in range(1, self.s):
            a = self.generateoffspring()
            if (random.random() < self.m):
                a.mutate()
            pop.append(a)

        return pop


