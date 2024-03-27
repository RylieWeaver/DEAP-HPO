import random
import os
import numpy as np
import random
from math import floor
import pickle
import uuid
from deap import base, creator, tools, algorithms
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import RMSprop

# Import all from the functions script
from functions import *



def create_individual_with_id():
    # Generate the genetic attributes
    attr_int = random.randint(50, 500)  # Number of neurons
    attr_float = random.uniform(0.0, 0.5)  # Dropout rate
    id = str(uuid.uuid4())  # Unique ID as a genetic attribute
    print(f"Creating individual with ID: {id}")
    
    # Create an individual with these attributes, including the unique ID
    return creator.Individual([attr_int, attr_float, id])


def create_toolbox():
    # Toolbox
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual_with_id)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", customMutate, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    return toolbox



def custom_evolutionary_process(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=True):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Make checkpoint
    save_checkpoint(population, filename=f"checkpoint.pkl")

    # Evaluate the entire population
    for ind in population:
        # Evaluate fitness
        fitness = toolbox.evaluate(ind)
        # Load latest checkpoint
        print(f"Loading checkpoint.pkl")
        checkpoint_data = load_checkpoint(filename=f"checkpoint.pkl")

        if checkpoint_data:
            loaded_population = checkpoint_data['population']
            # Find the individual in the loaded population by ID and update its fitness
            for loaded_ind in loaded_population:
                # Assuming the ID is the last element of the individual
                if loaded_ind[-1] == ind[-1]:  # Match based on the unique ID
                    print(f"Updating fitness for individual with ID: {ind[-1]}")
                    loaded_ind.fitness.values = fitness
                    break
            
            # Save checkpoint with the updated population
            save_checkpoint(loaded_population, filename=f"checkpoint.pkl")
            print(f"Saved checkpoint.pkl")
        else:
            print("Checkpoint data not found. Check the file path and try again.")

    print("---------------------FINISHED EVALUATING FIRST POPULATION---------------------")

    for generation in range(ngen):
        print(ngen)
        print(f"Generation {generation}")
        # Select the next generation individuals
        offspring = toolbox.select(population, mu)

        # Vary the pool of individuals
        offspring = algorithms.varOr(offspring, toolbox, lambda_, cxpb, mutpb)

        # Make checkpoint
        save_checkpoint(population, filename=f"checkpoint_{generation}.pkl", generation=generation, offspring=offspring)
        print(f"Saved checkpoint_{generation}.pkl")

        # Evaluate individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for i, ind in enumerate(invalid_ind):
            # Evaluate fitness
            fitness = toolbox.evaluate(ind)
            # Load latest checkpoint
            print(f"Loading checkpoint_{generation}.pkl")
            checkpoint_data = load_checkpoint(filename=f"checkpoint_{generation}.pkl")

            if checkpoint_data:
                loaded_population = checkpoint_data['population']
                generation = checkpoint_data['generation']
                if checkpoint_data['offspring']:
                    loaded_offspring = checkpoint_data['offspring']
                else:
                    loaded_offspring = offspring
                # Find the individual in the loaded population by ID and update its fitness
                for loaded_ind in loaded_offspring:
                    # Assuming the ID is the last element of the individual
                    if loaded_ind[-1] == ind[-1]:  # Match based on the unique ID
                        print(f"Updating fitness for individual with ID: {ind[-1]}")
                        loaded_ind.fitness.values = fitness
                        break

                # Save checkpoint with the updated population
                save_checkpoint(loaded_population, filename=f"checkpoint_{generation}.pkl", generation=generation, offspring=loaded_offspring)
                print(f"Saved checkpoint_{generation}.pkl")
            else:
                print("Checkpoint data not found. Check the file path and try again.")


        # Load latest checkpoint for updating the population and stats
        checkpoint_data = load_checkpoint(filename=f"checkpoint_{generation}.pkl")
        population = checkpoint_data['population']
        generation = checkpoint_data['generation']
        offspring = checkpoint_data['offspring']

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring (mu_plus_lambda strategy)
        merged_population = population + offspring
        population[:] = toolbox.select(merged_population, mu)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=generation, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook



# Main function
def main():
    # Create the DEAP toolbox
    toolbox = create_toolbox()

    # Attempt to load from checkpoint
    checkpoint_data = load_checkpoint(filename=f"checkpoint.pkl")
    if checkpoint_data:
        print("Checkpoint found")
        population = checkpoint_data['population']
    else:
        print("No checkpoint found")
        # Initialize things if no checkpoint is found
        population = toolbox.population(n=10)  # Example initialization
        # Save initial checkpoint
        save_checkpoint(population, filename=f"checkpoint.pkl")
        checkpoint_data = load_checkpoint(filename=f"checkpoint.pkl")
        population = checkpoint_data['population']

    # Number of individuals to select for the next generation (mu)
    mu = 10
    # Number of children to produce at each generation (lambda)
    lambda_ = 5
    # Probability of mating two individuals (crossover probability)
    cxpb = 0.2
    # Probability of mutating an individual (mutation probability)
    mutpb = 0.5

    # Statistics to gather
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Hall of Fame
    hof = tools.HallOfFame(1)

    # Run algorithm
    print(population)
    print("Finished initializing")
    pop, log = custom_evolutionary_process(population, toolbox, mu, lambda_, cxpb, mutpb, ngen=10, stats=stats, halloffame=hof, verbose=True)

    print("Finished evolving")

if __name__ == "__main__":
    main()




