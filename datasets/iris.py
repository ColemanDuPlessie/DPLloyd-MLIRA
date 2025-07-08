import numpy as np
import pandas as pd
from datasets.datasets import split_dataset
from kmeans import normalize

iris_file_path = "datasets/iris.data"

NUM_IRIS_SAMPLES = 150

def get_iris_dataset():
    """
    Load the iris dataset from a CSV file and return it as a numpy array.
    """
    df = pd.read_csv(iris_file_path, delimiter=",", header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
    df = df.reset_index(drop=True)
    return np.concatenate((df.to_numpy()[:, :-1].astype(np.float64), np.expand_dims(df.to_numpy()[:, -1], axis=1)), axis=1)

def split_iris_dataset(data, train_size=0.5):
    train, test = split_dataset(data, train_size=train_size)

    train_keys = np.array([0 if k == "Iris-setosa" else (1 if k == "Iris-versicolor" else 2) for k in train[:, -1]])
    train = normalize(train[:, :-1], (-1, 1)).astype(np.float64)
    test_keys = np.array([0 if k == "Iris-setosa" else (1 if k == "Iris-versicolor" else 2) for k in test[:, -1]])
    test = normalize(test[:, :-1], (-1, 1)).astype(np.float64)

    return train, train_keys, test, test_keys

if __name__ == "__main__":
    # Example usage
    iris_data = get_iris_dataset()
    print("Iris dataset loaded with shape:", iris_data.shape)
    print("First few rows of the dataset:\n", iris_data[:5])