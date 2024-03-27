import pickle

def load_from_pickle(filename):
    """Load and return the content of a pickle file."""
    try:
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"No such file: '{filename}'")
    except Exception as e:
        print(f"An error occurred while loading the pickle file: {e}")


def inspect_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    print(f"Type of loaded data: {type(data)}")
    if isinstance(data, list):
        print(f"List length: {len(data)}")
        print(f"Type of first element: {type(data[0]) if data else 'Empty list'}")
    else:
        print("Data is not a list.")
    return data



if __name__ == "__main__":

    # The path to your pickle file
    pickle_file_path = 'checkpoint_0.pkl'
    
    # Load the checkpoint file
    loaded_data = load_from_pickle(pickle_file_path)
    # loaded_data = inspect_pickle(pickle_file_path)
    
    # Check and print the loaded data
    if loaded_data is not None and 'population' in loaded_data:
        print("Data loaded from pickle file:")
        population = loaded_data['population']
        print(f"Population: {population}")
        for ind in population:
            fitness_values = getattr(ind, 'fitness', None).values if hasattr(ind, 'fitness') else 'No fitness'
            print(f"Individual: {ind}, Fitness: {fitness_values}")
    else:
        print("Failed to load data or population not found in the loaded data.")
