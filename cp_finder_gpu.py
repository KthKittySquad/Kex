import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from arhmm import arhmm_analysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from changepoints import compute_changepoints_for_babies

# Load the baby data from the pickle file
print("Opening data file...")

data_to_use = "baby_angles.pkl"
print(data_to_use)

with open(data_to_use, "rb") as f:
    baby_data = pickle.load(f)

if isinstance(baby_data, dict):
    baby_list = baby_data.values()
else:
    print("baby_data is not a dictionary")
    baby_list = baby_data

baby_pca_results = {}
print(f"Number of babies: {len(baby_list)}")
    
chunk_results = []

# Loop over each baby and perform PCA on their data individually
for idx, angles in enumerate(baby_list):
    scaler = StandardScaler()
    angles_scaled = scaler.fit_transform(angles)

    pca = PCA(n_components=7)  # Retain 95% of variance
    pca_result = pca.fit_transform(angles_scaled)

    baby_pca_results[idx] = {
        'pca_result': pca_result,
        'scaler': scaler,
        'pca': pca,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'components': pca.components_
    }

print("PCA processing completed for all babies.")

changepoint_params = {
    'k': 17,              # Significance level for changepoint detection
    'sigma': 3.5,         # Minimum segment size
    'peak_height': 0.5,   # Maximum segment size
    'peak_neighbors': 1,  # Number of neighbors to consider for peak detection
    'max_pcs': 7          # Maximum number of PCs to use
}

# PCA processing code completes and we compute changepoints for each baby for testing
# changepoint_average = compute_changepoints_for_babies(baby_pca_results, changepoint_params)

# PCA processing code completes and we compare changepoints across babies to moseq
print("Starting ARHMM model fitting")
for n in [2, 4, 8, 16, 32]:
    arhmm_analysis(baby_pca_results, output_dir="testing_output", n_states=n)