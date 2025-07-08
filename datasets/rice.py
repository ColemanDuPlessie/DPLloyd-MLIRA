import numpy as np
import pandas as pd
from datasets.datasets import split_dataset
from kmeans import normalize

rice_file_path = "datasets/Rice_Cammeo_Osmancik.xlsx"

NUM_RICE_SAMPLES = 3810

def get_rice_dataset():
    """
    Load the rice dataset from an Excel file and return it as a numpy array.
    """
    df = pd.read_excel(rice_file_path, sheet_name="Rice")
    df = df.reset_index(drop=True)
    return np.concatenate((df.to_numpy()[:, :-1].astype(np.float64), np.expand_dims(df.to_numpy()[:, -1], axis=1)), axis=1)

def split_rice_dataset(data, train_size=0.5):
    train, test = split_dataset(data, train_size=train_size)

    train_keys = np.array([0 if k == "Cammeo" else 1 for k in train[:, -1]])
    train = normalize(train[:, :-1], (-1, 1)).astype(np.float64)
    test_keys = np.array([0 if k == "Cammeo" else 1 for k in test[:, -1]])
    test = normalize(test[:, :-1], (-1, 1)).astype(np.float64)

    return train, train_keys, test, test_keys

if __name__ == "__main__":
    # Example usage
    rice_data = get_rice_dataset()
    print("Rice dataset loaded with shape:", rice_data.shape)
    print("First few rows of the dataset:\n", rice_data[:5])