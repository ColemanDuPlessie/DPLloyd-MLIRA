import numpy as np

def split_dataset(data, train_size=0.5):
    """
    Split a dataset into training and testing sets.
    
    Parameters:
    - data: The rice dataset as a numpy array.
    - train_size: Proportion of the dataset to include in the training set.
    
    Returns:
    - train_data: Training set as a numpy array.
    - test_data: Testing set as a numpy array.
    """
    np.random.shuffle(data)
    split_index = int(len(data) * train_size)
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data