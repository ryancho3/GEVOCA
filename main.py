
from Agent import Agent
from Population import Population
def print_hi(name):

    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.



if __name__ == '__main__':
    r = [0.35, 0.15, 0.4, 0.4, 0.3, 0.2, 0.3, 0.3, 0.4, 0.4, 0.3, 0.25]
    p = [0.037, 0.015, 0.02, 0.03, 0.03, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03, 0.01]
    K = 12
    T = 242
    size = 1000
    rate = 0.07
    a = Agent(K, T, r, p)
    a.initialize()
    print(a.genome)
    print(len(a.genome))
    print(a.expression)
    pop = Population(K, T, r, p, size, rate)
    pop.initialize()
    print(pop.agents[0].getgenome())

    for i in range(1000):
        pop.adjustall()
        pop.calcallfitness()
        best = pop.getbestagent()
        print("average fitness: " + str(pop.getaveragefitness()))
        print("best agent: " + str(best.expression))
        print("best agent fitness: " + str(best.fitness))
        pop.setpopulation(pop.nextgeneration())
