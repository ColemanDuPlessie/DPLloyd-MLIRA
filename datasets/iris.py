import numpy as np
import pandas as pd

iris_file_path = "datasets/iris.data"

NUM_SAMPLES = 3810

def get_iris_dataset():
    """
    Load the iris dataset from a CSV file and return it as a numpy array.
    """
    df = pd.read_csv(iris_file_path, delimiter=",", header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
    df = df.reset_index(drop=True)
    return np.concatenate((df.to_numpy()[:, :-1].astype(np.float64), np.expand_dims(df.to_numpy()[:, -1], axis=1)), axis=1)

if __name__ == "__main__":
    # Example usage
    iris_data = get_iris_dataset()
    print("Iris dataset loaded with shape:", iris_data.shape)
    print("First few rows of the dataset:\n", iris_data[:5])