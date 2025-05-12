# This script extracts the features from the .mat file and saves them to multiple .csv files for testing
import os
import glob
import numpy as np
import pandas as pd

np.random.seed(69) # Set a random seed for reproducibility

# Clean up existing CSV files
for csv_file in glob.glob('sample_data/*.csv'):
    os.remove(csv_file)

data = np.load('sample_data/testing_data.npz') # loading the testing data
test_inputs = data["test_inputs"] # Extracting the data from the weird matlab format
test_labels = data["test_labels"] # Extracting the data from the weird matlab format

rand_idx = (np.random.random(10)*len(test_inputs)).astype(int) # random indices to extract 10 random samples

# extracting the features and labels and saving them to .csv files
for i in rand_idx:
    features = pd.DataFrame(test_inputs[i].reshape(1, -1))
    # Convert label to integer and save
    label = pd.DataFrame({'label': [int(test_labels[i][0])]})
    features.to_csv(f'sample_data/features_{i}.csv', index=False)
    label.to_csv(f'sample_data/label_{i}.csv', index=False)
