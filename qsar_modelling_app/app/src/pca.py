import numpy as np
from src import preprocess as pp # Import the preprocess module to use the preprocess function

def pca (mat_file_path: str, n_pc: int = 12) -> np.ndarray:
    """
    Perform PCA on the QSAR data.
    
    Args:
        mat_file_path (str): Path to the .mat file containing the QSAR data.
        n_pc (int): Number of principal components to keep.
    
    Returns:
        np.ndarray: PCA-transformed data with labels appended.
    """
    features, labels = pp.preprocess(mat_file_path) # Preprocess the data and store the result in a variable

    cov_mat = np.cov(features.T) # Calculate the covariance matrix of the features

    eig_val, eig_vec = np.linalg.eigh(cov_mat) # Calculate the eigenvalues and eigenvectors of the covariance matrix

    srt_ord = np.argsort(eig_val)[::-1] # Sort the eigenvalues in descending order and get the indices

    srt_eig_val = eig_val[srt_ord] # Sort the eigenvalues in descending order
    srt_eig_vec = eig_vec[:, srt_ord] # Sort the eigenvectors according to the sorted eigenvalues

    pca_data = np.matmul(features, srt_eig_vec[:, :n_pc]) # Project the data onto the first n_pc principal components
    pca_data = np.append(pca_data, np.array([labels]).T, 1) # Append the labels to the PCA data

    np.savez('model_params/srt_eig_vec.npz', srt_eig_vec=srt_eig_vec)

    return pca_data # Return the PCA-transformed data with labels appended
