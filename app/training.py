# This file is used to train the model using the training data and save the model parameters and training MSE. It can be used to tweak essentially anything in the model training process.
# Just edit the variables below to change the training data, learning rate, number of epochs, and number of features. The model parameters and training MSE will be saved in a directory called 'model_params'.

import os
import numpy as np
from src import training_functions as trn
from src import pca

# loading the data from the pca function
inputs = pca.pca('data/QSAR_data(1).mat', 12)

# splitting the data into training and testing sets
split_idx = int(len(inputs)*0.8) # the classic 80/20 split

train_inputs = inputs[:split_idx, :-1]
train_labels = inputs[:split_idx, -1].reshape(-1, 1)  # Reshape to (n_samples, 1)

test_inputs = inputs[split_idx:, :-1]
test_labels = inputs[split_idx:, -1].reshape(-1, 1)  # Reshape to (n_samples, 1)

ALPHA = 0.01 # defining our learning rate alpha
N_EPOCHS = 100 # defining number of epochs
N_FEATURES = 12 # no of features = no of principal components, assignment done for readability 
wts, b = trn.init_wts(N_FEATURES) # initialising weights and biases
mse_mat = np.zeros(N_EPOCHS) # initialising a list to store training mse

if __name__ == "__main__": # this block is executed when the script is run directly
        
    for epoch in range(N_EPOCHS): # perform the iteration below n_epoch times
        wts, b, l, prediction = trn.iterate(train_inputs, train_labels, wts, b, ALPHA)

    os.makedirs('model_params', exist_ok=True) # creating a directory to store the model and mse data
    os.makedirs('sample_data', exist_ok=True)  # creating directory for test data

    np.savez('model_params/model.npz', wts=wts, b=b) # saving the model weights and biases in a .npz file
    np.savez('sample_data/testing_data.npz', test_inputs=test_inputs, test_labels=test_labels)
