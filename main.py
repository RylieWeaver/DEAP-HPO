import random
import os
import numpy as np
import random
from math import floor
import pickle
import uuid
from deap import base, creator, tools, algorithms
import tensorflow as tf

# Import all from the functions script
from functions import *



def create_individual_with_id():
    # Generate the genetic attributes
    attr_int = random.randint(50, 500)  # Number of neurons
    attr_float = random.uniform(0.0, 0.5)  # Dropout rate
    print(f"Creating individual with attributes: {attr_int}, {attr_float}")
    return creator.Individual([attr_int, attr_float])


def create_toolbox(mut_indpb, cx_indpb, tournsize):
    # Toolbox
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual_with_id)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxUniform, indpb=mut_indpb)
    toolbox.register("mutate", customMutate, indpb=cx_indpb)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    toolbox.register("evaluate", evaluate)

    return toolbox


def evolve(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, start_gen, stats=None, halloffame=None, verbose=True):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the population without fitness values
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    for ind in invalid_ind:
        # Evaluate fitness
        fitness = toolbox.evaluate(ind)
        # Assign fitnesses
        ind.fitness.values = fitness
        print("INDIVIDUAL FITNESS:", ind.fitness.values)

    print("---------------------FINISHED EVALUATING POPULATION---------------------")

    # Make checkpoint
    print("Saving checkpoint")
    save_checkpoint(population, filename=f"checkpoint.pkl")


    for generation in range(start_gen, ngen):
        print(f"Number of Generations: {ngen}")
        print(f"Executing Generation: {generation}")
        # Select the next generation individuals
        offspring = toolbox.select(population, mu)

        # Vary the pool of individuals
        offspring = algorithms.varOr(offspring, toolbox, lambda_, cxpb, mutpb)

        # Evaluate individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for i, ind in enumerate(invalid_ind):
            # Evaluate fitness
            fitness = toolbox.evaluate(ind)
            # Assign fitnesses
            ind.fitness.values = fitness
            print("INDIVIDUAL FITNESS:", ind.fitness.values)

        print("---------------------FINISHED EVALUATING OFFSPRING---------------------")

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring (mu_plus_lambda strategy)
        merged_population = population + offspring
        population[:] = toolbox.select(merged_population, mu)

        # Print the fitness values of the population
        for ind in population:
            print(ind.fitness.values)

        # Make checkpoint
        save_checkpoint(population, filename=f"checkpoint.pkl", generation=generation)
        print(f"Saved Checkpoint")

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=generation, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook



# Main function
def main():
    # Genetic Algorithm parameters
    population_size = 10
    num_generations = 10
    offspring_proportion = 0.5
    mut_prob = 0.8
    cx_prob = 0.2
    mut_indpb = 0.8
    cx_indpb = 0.2
    tournsize = 3

    # Create the DEAP toolbox
    toolbox = create_toolbox(mut_indpb, cx_indpb, tournsize)

    # Attempt to load from checkpoint
    checkpoint_data = load_checkpoint(filename=f"checkpoint.pkl")
    if checkpoint_data:
        print("Checkpoint found")
        population = checkpoint_data['population']
        generation = checkpoint_data['generation']
        print("Loaded checkpoint data:")
        for ind in population:
            # This assumes ind is an Individual object and has a fitness attribute
            print(f"Individual: {ind}, Fitness: {ind.fitness.values}")
        print(f"Generation: {generation}")
        print("")
    else:
        print("No checkpoint found")
        # Initialize things if no checkpoint is found
        print("Initializing population")
        population = toolbox.population(n=10)  # Example initialization
        # Save initial checkpoint
        save_checkpoint(population, filename=f"checkpoint.pkl")
        checkpoint_data = load_checkpoint(filename=f"checkpoint.pkl")
        population = checkpoint_data['population']
        generation = checkpoint_data['generation']
        print("Saved checkpoint data:")
        print(f"Population: {population}")
        print(f"Generation: {generation}")
        print("")


    # Statistics to gather
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Hall of Fame
    hof = tools.HallOfFame(1)

    # Run algorithm
    print("Finished initializing")
    population, logbook = evolve(population, toolbox, population_size, floor(population_size * offspring_proportion), cx_prob, mut_prob, num_generations, generation, stats=stats, halloffame=hof)

    print("Finished evolving")

if __name__ == "__main__":
    main()
