# QSAR Modelling using Machine Learning

This repository contains a QSAR (Quantitative Structure-Activity Relationship) modelling project that uses a Single Layer Perceptron (SLP) neural network to predict the biodegradability of chemical compounds based on their structural features.

## Overview

QSAR models predict the biological activity of chemical compounds based on their structural properties. This project implements a modern machine learning approach using a Single Layer Perceptron, which offers several advantages:
- Efficient computation compared to traditional methods like SVM or Random Forest
- Linear relationship modeling between features and biodegradability
- Fast prediction times suitable for real-time applications

## Project Structure

```
app/
├── app.py              # Main Flask application
├── training.py         # Model training script
├── extract_features.py # Feature extraction utilities
├── requirements.txt    # Project dependencies
├── data/              # Training data directory
├── model_params/      # Saved model parameters
├── sample_data/       # Sample data for testing
├── src/              # Source code utilities
└── templates/        # Web application templates
```

## Features

- Web-based interface for making predictions
- Pre-trained model for immediate use
- Sample data for testing
- Feature extraction from chemical structures
- PCA-based dimensionality reduction
- Real-time prediction capabilities

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r app/requirements.txt
```

## Usage

1. Start the web application:
```bash
python app/app.py
```

2. Access the web interface at `http://localhost:5000`

3. Use the sample data or upload your own chemical compound features for prediction

## Model Training

The model can be retrained using the `training.py` script:

```bash
python app/training.py
```

Training parameters can be modified in the script:
- Learning rate (ALPHA)
- Number of epochs (N_EPOCHS)
- Number of features (N_FEATURES)

## Technical Details

- The model uses a Single Layer Perceptron with sigmoid activation
- Features are preprocessed using PCA for dimensionality reduction
- Training uses an 80/20 split for training/testing data
- Model parameters are saved in NumPy format for easy loading

## Dependencies

- Flask 3.0.2
- NumPy 1.26.4
- SciPy 1.12.0
- pandas 2.2.1
- python-dotenv 1.0.1

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## How It Works:
QSAR models are essentially models which predict activities of chemicals using observed structure historically a chemist was needed to do this job, then came rule based models and now in the modern age we can leverage machine learning to do the same, early qsar models using support vector machines and random forest which we already know come with a large computational overhead. and it is in these places where our single layer perceptrons can truly shine because it is essentially deriving a linear relationship between the models.

So now coming to the data set itself. The data set was provided 