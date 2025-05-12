# This script preprocesses the data for the biodegradability predictor.
# It loads the data from a .mat file, converts it to a numpy array, stores its dimensions if needed.

# Import necessary libraries
from scipy.io import loadmat
import numpy as np


def preprocess(mat_file_path: str, random_seed=69) -> tuple:
    """
    Preprocess QSAR data from a .mat file.
    Args:
        mat_file_path (str): Path to the .mat file containing the QSAR data.
        random_seed (int): Random seed for reproducibility.
    
    Returns:
        tuple: A tuple containing the standardized features and labels.
    """
    np.random.seed(random_seed) # Set a random seed for reproducibility

    data = loadmat(mat_file_path) # Load the .mat file using scipy.io.loadmat
    data = data["QSAR_data"] # Extracting the data from the weird matlab format

    data = np.array(data) # Convert the data to a numpy array if it is not already
    np.random.shuffle(data) # Shuffle the data remove label assymmetry

    labels = data[:, -1] # Extract the labels from the last column
    features = data[:, :-1] # Extract the features from the rest of the columns

    std_ft = (features - features.mean(0)) / features.std(0) # Standardize the features
    # Standardization is a common preprocessing step in machine learning to ensure that all features have the same scale.

    return (std_ft, labels) # Return the standardized features and labels
