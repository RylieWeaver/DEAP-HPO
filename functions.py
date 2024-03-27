import random
import os
import sys
import numpy as np
import random
from math import floor
import pickle
from deap import base, creator, tools, algorithms
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import RMSprop


def get_data():
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize data
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255

    # Convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Subset Data
    num_train_samples = 10000
    num_test_samples = 2500

    x_train = x_train[:num_train_samples]
    y_train = y_train[:num_train_samples]

    x_test = x_test[:num_test_samples]
    y_test = y_test[:num_test_samples]

    return x_train, y_train, x_test, y_test


def save_checkpoint(population, filename="checkpoint.pkl", generation=None, offspring=None):
    """
    Save the state of the evolutionary algorithm to a checkpoint file. Can save the entire population or individual evaluations.

    Parameters:
    - population: The current population of individuals.
    - generation: The current generation number.
    - evaluated_individuals: List of booleans indicating which individuals have been evaluated.
    - individual_index: Index of the individual being evaluated (for individual checkpoints).
    - state: "population" for population checkpoints, "pre_eval" or "post_eval" for individual checkpoints.
    - filename: The base name of the file to save the checkpoint to.
    """
    checkpoint_data = {
    "population": population,  # Ensure 'population' is the variable containing your individuals
    "generation": generation,
    "rndstate": random.getstate()
    }
    # Include offspring in the checkpoint only if it's provided
    if offspring is not None:
        checkpoint_data['offspring'] = offspring
    else:
        checkpoint_data['offspring'] = None

    with open(filename, "wb") as cp_file:
        pickle.dump(checkpoint_data, cp_file)


def load_checkpoint(filename="checkpoint.pkl"):
    """
    Load the evolutionary algorithm's state from a checkpoint file.

    Parameters:
    - filename: The name of the file from which to load the checkpoint.

    Returns:
    - The checkpoint data, including the population, generation number, and random state,
      or None if the checkpoint file does not exist.
    """
    if os.path.isfile(filename):
        with open(filename, "rb") as cp_file:
            checkpoint_data = pickle.load(cp_file)
            return checkpoint_data
    else:
        print(f"No such file: '{filename}'")
        return None


def evaluate(individual):
    # Unpack individual's genes (hyperparameters)
    # print(individual)
    n_neurons, dropout_rate, id = individual

    # Build neural network model
    model = Sequential([
        Input(shape=(784,)),  # Specify input shape here
        Dense(n_neurons, activation='relu'),
        Dropout(dropout_rate),
        Dense(10, activation='softmax')
    ])


    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    # Train model
    x_train, y_train, x_test, y_test = get_data()
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=5,
              verbose=0,
              validation_data=(x_test, y_test))

    # Evaluate model
    score = model.evaluate(x_test, y_test, verbose=0)
    return score[1],  # Return accuracy as fitness


def customMutate(individual, mu=0, sigma=1, indpb=0.2):
    """
    Custom mutation function that applies Gaussian mutation but rounds
    the neuron count (first gene) while leaving the dropout rate (second gene) as is.

    Parameters:
    - individual: The individual to mutate.
    - mu: Mean for the Gaussian mutation.
    - sigma: Standard deviation for the Gaussian mutation.
    - indpb: Independent probability for each attribute to be mutated.
    """
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            if i == 0:  # Assuming the first gene is the neuron count
                neuron_sigma = 20
                individual[i] += floor(random.gauss(mu, neuron_sigma))
                # Ensure the neuron count stays within desired bounds, for example:
                individual[i] = max(1, min(individual[i], 1000))
            elif i == 1:  # Assuming the second gene is the dropout rate
                individual[i] += random.gauss(mu, sigma)
                # Ensure the dropout rate stays within [0, 1]
                individual[i] = max(0, min(individual[i], 1))
            elif i == 2:
                pass
    return individual,


