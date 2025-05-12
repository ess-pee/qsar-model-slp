import os
from glob import glob
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd

app = Flask(__name__)

def predict(i, weights, biases):
    # Ensure inputs are numpy arrays and have correct shapes
    i = np.array(i, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    biases = np.array(biases, dtype=np.float64)
    
    # Reshape if necessary
    if len(i.shape) == 1:
        i = i.reshape(1, -1)
    if len(weights.shape) == 1:
        weights = weights.reshape(-1, 1)
    
    pred = sigmoid(np.dot(i, weights) + biases)
    return pred

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Load model parameters
params = np.load('model_params/model.npz')
wts = params['wts']
b = params['b']

def get_sample_files():
    """Get list of sample feature files and their corresponding labels"""
    feature_files = sorted(glob('sample_data/features_*.csv'))
    sample_files = []
    for feature_file in feature_files:
        idx = feature_file.split('_')[-1].split('.')[0]
        label_file = f'sample_data/label_{idx}.csv'
        if os.path.exists(label_file):
            sample_files.append({
                'features': feature_file,
                'label': label_file,
                'index': idx
            })
    return sample_files

@app.route('/')
def home():
    sample_files = get_sample_files()
    return render_template('index.html', sample_files=sample_files, pd=pd)

@app.route('/predict/<int:index>')
def make_prediction(index):
    try:
        # Load features and label
        features_df = pd.read_csv(f'sample_data/features_{index}.csv')
        label_df = pd.read_csv(f'sample_data/label_{index}.csv')
        
        features = features_df.values[0]
        actual_label = label_df['label'].values[0]
        
        # Make prediction
        prediction = int(np.round(predict(features, wts, b)))
        
        return jsonify({
            'prediction': prediction,
            'actual': int(actual_label),
            'features': features.tolist(),
            'message': 'Prediction successful'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Error making prediction'
        }), 400

if __name__ == '__main__':
    app.run(debug=True)
