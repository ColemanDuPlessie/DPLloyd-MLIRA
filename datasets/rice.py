import numpy as np
import pandas as pd

rice_file_path = "Rice_Cammeo_Osmancik.xlsx"

NUM_SAMPLES = 3810

def get_rice_dataset():
    """
    Load the rice dataset from an Excel file and return it as a numpy array.
    """
    df = pd.read_excel(rice_file_path, sheet_name="Rice")
    df = df.reset_index(drop=True)
    return np.concatenate((df.to_numpy()[:, :-1].astype(np.float64), np.expand_dims(df.to_numpy()[:, -1], axis=1)), axis=1)

if __name__ == "__main__":
    # Example usage
    rice_data = get_rice_dataset()
    print("Rice dataset loaded with shape:", rice_data.shape)
    print("First few rows of the dataset:\n", rice_data[:5])