import random, numpy
from deap import algorithms, base, creator, tools

GENERATIONS = 30 
POPULATION_SIZE = 20

MIN_SPEED = 0
MAX_SPEED = 50

def runCoSimulation(EGO_speed, GVT_speed):
    # Here the CoSimulation should be used to produce the value that should be optimized (e.g., TTC)
    return EGO_speed + GVT_speed

def evaluate(individual):
    result = runCoSimulation(individual[0], individual[1])
    return result,

def mutUniform(individual, low, up, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.uniform(low, up)
    return individual,

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("randomUniform", random.uniform, MIN_SPEED, MAX_SPEED)
toolbox.register("initIndividual", tools.initRepeat, creator.Individual, toolbox.randomUniform, 2)
toolbox.register("initPopulation", tools.initRepeat, list, toolbox.initIndividual, POPULATION_SIZE)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxUniform, indpb=0.4)
toolbox.register("mutate", mutUniform, low=MIN_SPEED, up=MAX_SPEED, indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)

# start of the actual program
pop = toolbox.initPopulation()
hof = tools.HallOfFame(5)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean)
stats.register("std", numpy.std)
stats.register("min", numpy.min)
stats.register("max", numpy.max)

pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.2, mutpb=0.1, ngen=GENERATIONS, stats=stats, halloffame=hof, verbose=True)

print(hof)