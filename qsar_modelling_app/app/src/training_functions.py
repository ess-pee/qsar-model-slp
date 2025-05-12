import numpy as np

def sigmoid(x): # sigmoid activation
    
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x): # derivative of sigmoid
    
    return x * (1 - x)

def init_wts(input_features): # initialise weights between -1 and 1
    
    wts = 2 * np.random.random((input_features,1)) - 1
    bs = 2 * np.random.random((1,1)) - 1

    return wts, bs

def predict(i, weights, biases): # generate predictions / feedforward
    
    prediction = sigmoid(np.dot(i, weights) + biases)

    return prediction

def update_weights(o, predictions, weights, biases, i, alpha): # weight adjustment of the perceptron / backpropagation

    loss = o - predictions
    adjustments = loss * sigmoid_derivative(predictions)
    weights += alpha * np.dot(i.T, adjustments)
    biases += alpha * np.sum(adjustments,0)

    return weights, biases, loss

def iterate(inputs, outputs, wts, b, learning_rate): # just a function to combine everything above into 1

    prediction = predict(inputs, wts, b)
    wts, b, l = update_weights(outputs,prediction,wts,b,inputs, learning_rate)
    
    return wts, b, l, prediction
