import os
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.signal
from tqdm import tqdm
from copy import deepcopy
from scipy.stats import mode
from scipy.ndimage import gaussian_filter1d

def gauss_smooth(y, sigma):
    """
    Apply gaussian smoothing to a 1D array.
    
    Args:
        y (numpy.ndarray): Array to smooth
        sigma (float): Standard deviation of gaussian kernel
        
    Returns:
        numpy.ndarray: Smoothed array
    """
    
    return gaussian_filter1d(y, sigma)

def get_changepoints(scores, k=5, sigma=3.5, peak_height=.5, peak_neighbors=1,
                     baseline=True):
    """
    Compute changepoints and its corresponding distribution. Changepoints describe
    the magnitude of frame-to-frame changes in pose.

    Args:
    scores (numpy.ndarray): Principal component scores, shape (n_frames, n_components)
    k (int): Lag to use for derivative calculation.
    sigma (int): Standard deviation of gaussian smoothing filter.
    peak_height (float): Minimum height of peaks to be considered changepoints.
    peak_neighbors (int): Number of neighboring points to compare for finding peaks.
    baseline (bool): Whether to normalize data by subtracting minimum.

    Returns:
    cps (numpy.ndarray): Array of changepoint indices
    """
    if type(k) is not int:
        k = int(k)

    if type(peak_neighbors) is not int:
        peak_neighbors = int(peak_neighbors)

    normed_df = deepcopy(scores)
    
    nanidx = np.isnan(normed_df)
    normed_df[nanidx] = 0

    # apply gaussian smoothing if sigma is specified
    if sigma is not None and sigma > 0:
        for i in range(scores.shape[1]):
            normed_df[:, i] = gauss_smooth(normed_df[:, i], sigma)

    # Compute squared differences for each component (using lag k)
    # This calculates frame-to-frame changes
    squared_diffs = np.zeros_like(normed_df)
    for i in range(normed_df.shape[1]):
        component = normed_df[:, i]
        padded = np.pad(component, (k, k), mode='edge')
        for j in range(len(component)):
            squared_diffs[j, i] = (padded[j+k] - padded[j])**2

    # Replace original normed_df with squared differences
    normed_df = squared_diffs
    
    # Restore NaN values
    normed_df[nanidx] = np.nan
    
    # Mask out the edges affected by smoothing
    if sigma > 0:
        edge_size = int(6 * sigma)
        if normed_df.shape[0] > 2 * edge_size:
            normed_df[:edge_size, :] = np.nan
            normed_df[-edge_size:, :] = np.nan

    # Calculate mean across components (axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        normed_df = np.nanmean(normed_df, axis=1)

        if baseline:
            min_val = np.nanmin(normed_df)
            if not np.isnan(min_val):
                normed_df -= min_val

        normed_df = np.squeeze(normed_df)
        peaks = scipy.signal.find_peaks(normed_df, height=peak_height, distance=peak_neighbors)[0]
        
    return peaks

def compute_changepoints_for_babies(baby_pca_results, changepoint_params, output_dir='./outputs'):
    """
    Compute changepoints for baby movement data based on PCA results.
    
    Args:
        baby_pca_results (dict): Dictionary containing PCA results for each baby
        changepoint_params (dict): Parameters for changepoint detection
        output_dir (str): Directory to save output files
    
    Returns:
        dict: Dictionary containing changepoints for each baby
    """
   
    os.makedirs(output_dir, exist_ok=True)
    
    all_changepoints = {}
    
    # For each baby, compute changepoints with progress bar
    for baby_idx in tqdm(baby_pca_results.keys(), desc="Processing babies", unit="baby"):
        baby_data = baby_pca_results[baby_idx]
        pca_result = baby_data['pca_result']
        
        n_pcs = min(pca_result.shape[1], changepoint_params['max_pcs'])
        pca_components = pca_result[:, :n_pcs]
        
        # Compute changepoints
        changepoints = get_changepoints(
            pca_components, 
            k=changepoint_params.get('k', 15),
            sigma=changepoint_params.get('sigma', 3.5),
            peak_height=changepoint_params.get('peak_height', 0.5),
            peak_neighbors=changepoint_params.get('peak_neighbors', 1),
        )
        
        all_changepoints[baby_idx] = changepoints 
    
    all_block_durations = []
        
    for baby_idx, cps in all_changepoints.items():
        if len(cps) > 1:
            block_durations = np.diff(cps)
            all_block_durations.extend(block_durations) 
    
    if all_block_durations:
        block_durs = np.array(all_block_durations)
        
        # Sanitize data by only keeping values within 3 standard deviations
        mean_dur = np.mean(block_durs)
        std_dur = np.std(block_durs)
        lower_bound = mean_dur - 2 * std_dur
        upper_bound = mean_dur + 2 * std_dur
        
        # Filter data to keep only values within bounds
        block_durs_filtered = block_durs[(block_durs >= lower_bound) & (block_durs <= upper_bound)]
        
        # Print statistics before and after filtering
        print(f"Original data: {len(block_durs)} points, mean={mean_dur:.2f}, std={std_dur:.2f}")
        print(f"Filtered data: {len(block_durs_filtered)} points, " 
              f"removed {len(block_durs) - len(block_durs_filtered)} outliers")
        print(f"Removed {round(((len(block_durs) - len(block_durs_filtered))/len(block_durs))*100)}% of data points as outliers")
        
        max_block = int(np.ceil(upper_bound))
        
        # Create and save a histogram with KDE curve overlay
        plt.figure(figsize=(10, 6))
        
        # Plot histogram
        plt.hist(block_durs_filtered, bins=np.arange(0, max_block+2)-0.5, alpha=0.75, density=True, label='Histogram')
        
        # Add KDE curve
        kde = sns.kdeplot(block_durs_filtered, color='blue', linewidth=1, label='KDE')
        
        plt.xlabel('Block Duration')
        plt.ylabel('Density')
        plt.title('Distribution of Block Durations (Within 3 std)')
        plt.legend()
        plt.savefig(f'{output_dir}/changepoint_distribution_sanitized.png')
        plt.close()
        
    else:
        print("No block durations found to plot. Check if changepoints are being detected correctly.")
    
    return mean_dur