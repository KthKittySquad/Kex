"""
AR-HMM Analysis for Baby Data
Implementation using dynamax library
"""
import os
import jax
import time
import pickle
import psutil
import tempfile
import jax.tree
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jr
import seaborn as sns
from itertools import count
from tqdm.auto import tqdm
from jax import tree_util
from time import time
from functools import partial
from dynamax.hidden_markov_model import LinearAutoregressiveHMM

def prepare_data(baby_pca_results):
    """
    Prepare baby data for AR-HMM training with Dynamax.
    
    Args:
        baby_pca_results (dict): Dictionary of PCA results for each baby
    
    Returns:
        tuple: Processed data components including JAX arrays and indices
    """
    # Extract data dimensions from the first baby's PCA results
    first_baby_key = list(baby_pca_results.keys())[0]
    first_baby_data = baby_pca_results[first_baby_key]['pca_result']
    obs_dim = first_baby_data.shape[1]  # Dimension of PCA components
    
    all_data_sequences = []
    baby_indices = []
    
    for baby_id, baby_data in baby_pca_results.items():
        # Get the PCA result array for this baby
        pca_data = baby_data['pca_result']
        
        data_array = np.asarray(pca_data, dtype=np.float32)
        
        # Convert to JAX array
        data_array_jax = jnp.array(data_array)
        
        all_data_sequences.append(data_array_jax)
        baby_indices.append(baby_id)
    
    print(f"Number of sequences: {len(all_data_sequences)}")
    print(f"Example sequence shape: {all_data_sequences[0].shape}")
    
    return all_data_sequences, baby_indices, obs_dim

def create_arhmm_model(arhmm_params, obs_dim):
    """
    Create AR-HMM model with dynamax library.
    
    Args:
        arhmm_params (dict): AR-HMM model configuration parameters
        obs_dim (int): Observation dimension
    
    Returns:
        tuple: Model, initial parameters, and parameter properties
    """
    n_states = arhmm_params.get("n_states", 5)
    ar_lags = arhmm_params.get("ar_lags", 1)
    
    model = LinearAutoregressiveHMM(
        num_states=n_states,
        emission_dim=obs_dim,
        num_lags=ar_lags
    )
    
    key = jr.PRNGKey(arhmm_params.get("seed", 0))
    keys = jr.split(key, 10)
    
    # Initialize model parameters
    transition_matrix = arhmm_params.get("transition_matrix", 
                                        0.95 * jnp.eye(n_states) + 0.05 / (n_states - 1) * (1 - jnp.eye(n_states)))
    
    # Default
    params, param_props = model.initialize(
        key=keys[0],
        method="prior",
        transition_matrix=transition_matrix
    )
    
    return model, params, param_props

def train_model_with_dynamax(model, params, param_props, all_data_sequences_jax, arhmm_params):
    """
    Train AR-HMM model on concatenated data with tracking of original sequence boundaries.
    Optimized for better performance and memory management.
    
    Args:
        model: Dynamax ARHMM model instance
        params: Initial model parameters
        param_props: Parameter properties
        all_data_sequences_jax: List of JAX arrays containing sequence data
        arhmm_params: Dictionary of model hyperparameters
        
    Returns:
        tuple: (trained_params, log_probs, sequence_boundaries)
    """

    
    max_em_iters = arhmm_params.get("max_em_iters", 100)
    em_tol = arhmm_params.get("em_tol", 1e-4)
    
    max_sequences = arhmm_params.get("max_training_sequences", None)
    
    training_sequences = all_data_sequences_jax
    if max_sequences is not None and max_sequences < len(all_data_sequences_jax):
        print(f"Using first {max_sequences} sequences out of {len(all_data_sequences_jax)} for training")
        training_sequences = all_data_sequences_jax[:max_sequences]
    else:
        print(f"Using all {len(all_data_sequences_jax)} sequences for training")
    
    total_data_points = sum(seq.shape[0] for seq in training_sequences)
    print(f"Total data points for training: {total_data_points}")
    
    # Mem usage
    try:        
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)
        print(f"Memory usage before concatenation: {mem_before:.1f} MB")
    except ImportError:
        print("Install psutil for memory tracking")
    
    start_time = time.time()
    lengths = [seq.shape[0] for seq in training_sequences]
    sequence_boundaries = [0]
    for length in lengths:
        sequence_boundaries.append(sequence_boundaries[-1] + length)
    
    print(f"Sequence boundary calculation: {time.time() - start_time:.2f} seconds")
    print(f"Number of sequences: {len(sequence_boundaries) - 1}")
    
    print("Concatenating sequences...")
    concat_start_time = time.time()
    concat_sequence = jnp.concatenate(training_sequences, axis=0)
    concat_time = time.time() - concat_start_time
    print(f"Concatenation completed in {concat_time:.2f} seconds")
    print(f"Concatenated sequence shape: {concat_sequence.shape}")
    
    # Track memory after concatenation
    try:
        mem_after = process.memory_info().rss / (1024 * 1024)
        print(f"Memory usage after concatenation: {mem_after:.1f} MB (increase: {mem_after - mem_before:.1f} MB)")
    except:
        pass
    
    print("Computing inputs...")
    inputs_start_time = time.time()
    inputs = model.compute_inputs(concat_sequence)
    inputs_time = time.time() - inputs_start_time
    print(f"Input computation completed in {inputs_time:.2f} seconds")
    
    # Configure EM training parameters
    em_kwargs = {
        'params': params,
        'props': param_props,
        'emissions': concat_sequence,
        'inputs': inputs,
        'num_iters': max_em_iters,
        'verbose': True
    }
    
    # Add progress tracking
    print("Starting EM training...")
    print(f"Max iterations: {max_em_iters}, Convergence tolerance: {em_tol}")
    
    train_start_time = time.time()
    trained_params, log_probs = model.fit_em(**em_kwargs)
    train_time = time.time() - train_start_time
    
    # Calculate training statistics
    num_iters = len(log_probs)
    if num_iters > 1:
        improvement = log_probs[-1] - log_probs[0]
        improvement_per_iter = improvement / (num_iters - 1)
    else:
        improvement = 0
        improvement_per_iter = 0
    
    print(f"Training completed in {train_time:.2f} seconds ({train_time/num_iters:.2f} seconds per iteration)")
    print(f"Iterations: {num_iters}/{max_em_iters}")
    print(f"Final log-likelihood: {log_probs[-1]:.2f}")
    print(f"Total improvement: {improvement:.2f} ({improvement_per_iter:.2f} per iteration)")
    
    if num_iters < max_em_iters:
        print("Early stopping: Convergence achieved!")
    else:
        print("Maximum iterations reached. Check if model has converged.")
    
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(log_probs, '-o')
        plt.title('Log Likelihood Progression')
        plt.xlabel('EM Iteration')
        plt.ylabel('Log Likelihood')
        plt.grid(True)
        
        temp_dir = tempfile.gettempdir()
        plot_path = os.path.join(temp_dir, 'log_likelihood_progression.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Log-likelihood plot saved to {plot_path}")
    except Exception as e:
        print(f"Could not create log-likelihood plot: {e}")
    
    return trained_params, log_probs, sequence_boundaries

def infer_states(model, params, all_data_sequences_jax, baby_indices, sequence_boundaries=None):
    """
    Infer most likely states for each baby and calculate log-likelihoods simultaneously,
    with optimized performance using JAX's parallelization capabilities.
    
    Args:
        model: Trained dynamax ARHMM model
        params: Trained model parameters
        all_data_sequences_jax: Processed baby data sequences
        baby_indices: List of baby indices
        sequence_boundaries: List of indices where each sequence starts in concatenated data
    
    Returns:
        tuple: (baby_states, log_likelihoods) - Dictionary of state sequences and dictionary of log-likelihoods
    """

    baby_states = {}
    log_likelihoods = {}
    normalized_log_likelihoods = {}
    
    if sequence_boundaries is not None and len(sequence_boundaries) > 1:
        num_sequences = len(sequence_boundaries) - 1  
    else:
        num_sequences = len(baby_indices)
    
    print(f"\nInferring states and calculating log-likelihoods for {num_sequences} babies:")
    
    processed = 0
    start_time = time()
    
    # Create JIT-compiled version
    @partial(jax.jit, static_argnums=(0,))
    def compute_for_sequence(model, params, sequence):
        """JIT-compiled function to compute states and log-likelihood for a single sequence"""
        inputs = model.compute_inputs(sequence)
        most_likely_states = model.most_likely_states(params, sequence, inputs=inputs)
        log_lik = model.marginal_log_prob(params, sequence, inputs=inputs)
        return most_likely_states, log_lik
    
    # Process in small batches, can be adjusted
    batch_size = 10
    
    for batch_start in range(0, num_sequences, batch_size):
        batch_end = min(batch_start + batch_size, num_sequences)
        batch_indices = range(batch_start, batch_end)
        
        # Process each sequence in the batch
        for i in batch_indices:
            try:
                baby_id = baby_indices[i]
                baby_data = all_data_sequences_jax[i]
                
                # Skip sequences that are too short for the AR model
                if len(baby_data) <= model.num_lags:
                    print(f"\rSkipping baby {baby_id}: sequence too short for AR model with {model.num_lags} lags")
                    baby_states[baby_id] = None
                    continue
                
                most_likely_states, log_lik = compute_for_sequence(model, params, baby_data)
                baby_states[baby_id] = np.array(most_likely_states)
                log_likelihoods[baby_id] = float(log_lik)
                
                # Normalize by sequence length
                seq_length = baby_data.shape[0]
                normalized_log_likelihoods[baby_id] = float(log_lik) / seq_length
                
            except Exception as e:
                print(f"\rFailed to process baby {baby_id}: {e}")
                baby_states[baby_id] = None
            
            # Update progress
            processed += 1
            if processed % 5 == 0 or processed == num_sequences:
                progress = int(50 * processed / num_sequences)
                percent = int(100 * processed / num_sequences)
                elapsed = time() - start_time
                remaining = (elapsed / processed) * (num_sequences - processed) if processed > 0 else 0
                
                print(f"\r[{'#' * progress}{'_' * (50-progress)}] {percent}% - {processed}/{num_sequences} " +
                      f"({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)", end="")
    
    total_time = time() - start_time
    print(f"\nProcessing completed in {total_time:.2f} seconds ({total_time/num_sequences:.2f}s per sequence)")
    
    valid_states = [states for states in baby_states.values() if states is not None]
    if valid_states:
        avg_length = sum(len(states) for states in valid_states) / len(valid_states)
        print(f"Average sequence length: {avg_length:.1f} frames")
    
    if log_likelihoods:
        avg_ll = sum(log_likelihoods.values()) / len(log_likelihoods)
        print(f"Average log-likelihood: {avg_ll:.1f}")
        
    return baby_states, log_likelihoods, normalized_log_likelihoods

def save_models(model, params, models_dir):
    """
    Save trained models to disk.
    
    Args:
        model: Dynamax ARHMM model
        params: Model parameters
        models_dir: Directory to save models
    
    Returns:
        tuple: Paths to saved models
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Save dynamax model and parameters
    model_path = os.path.join(models_dir, "arhmm_model.pkl")
    params_path = os.path.join(models_dir, "arhmm_params.pkl")
    
    try:
        numpy_params = jax.tree.map(lambda x: np.array(x), params)
    except (ImportError, AttributeError):
        numpy_params = tree_util.tree_map(lambda x: np.array(x), params)
    
    with open(model_path, 'wb') as f:
        pickle.dump(LinearAutoregressiveHMM, f)
    
    with open(params_path, 'wb') as f:
        pickle.dump(numpy_params, f)
    
    return model_path, params_path

def plot_individual_states(baby_states, output_dir="output/state_plots"):
    """
    Create individual state sequence plots for each baby and save to output folder.
    
    Args:
        baby_states (dict): State sequences for each baby
        output_dir (str): Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    sns.set_context("notebook")
    
    # Plot each baby's state sequence
    for baby_id, states in baby_states.items():
        if states is not None:
            plt.figure(figsize=(12, 3))
            plt.imshow(states[None, :], aspect='auto', cmap='viridis')
            plt.colorbar(label='State')
            plt.title(f'State Sequence for Baby {baby_id}')
            plt.xlabel('Time')
            plt.ylabel('States')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/baby_{baby_id}_states.png", dpi=300)
            plt.close()
    
    print(f"Individual state plots saved to {output_dir}/")

def plot_stacked_states(baby_states, baby_indices, output_dir="output/state_plots"):
    """
    Create stacked visualization of state sequences across babies and save to output folder.
    
    Args:
        baby_states (dict): State sequences for each baby
        baby_indices (list): List of baby indices
        output_dir (str): Directory to save the plots
    """

    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out babies with missing states
    valid_babies = [baby_id for baby_id in baby_indices if baby_id in baby_states and baby_states[baby_id] is not None]
    
    if not valid_babies:
        print("No valid state sequences to plot.")
        return
    
    # Create stacked visualization
    plt.figure(figsize=(14, 8))
    
    n_babies = len(valid_babies)
    max_len = max([len(baby_states[baby_id]) for baby_id in valid_babies])
    
    stacked_matrix = np.ones((n_babies, max_len)) * np.nan
    
    for i, baby_id in enumerate(valid_babies):
        seq_len = len(baby_states[baby_id])
        stacked_matrix[i, :seq_len] = baby_states[baby_id]
    
    # Plot the stacked states
    im = plt.imshow(stacked_matrix, aspect='auto', cmap='viridis', interpolation='none')
    plt.colorbar(im, label='State')
    plt.title('Stacked State Sequences Across Babies')
    plt.xlabel('Time')
    plt.ylabel('Baby Index')
    plt.yticks(range(n_babies), valid_babies)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/stacked_states.png", dpi=300)
    plt.close()
    
    print(f"Stacked state plot saved to {output_dir}/stacked_states.png")

def plot_state_usage(baby_states, baby_indices, n_states, output_dir="output/state_plots"):
    """
    Visualize state usage distribution across babies and save to output folder.
    
    Args:
        baby_states (dict): State sequences for each baby
        baby_indices (list): List of baby indices
        n_states (int): Number of states
        output_dir (str): Directory to save the plots
    
    Returns:
        dict: State usage proportions for each baby
    """

    os.makedirs(output_dir, exist_ok=True)
    
    state_usage = {}
    
    for baby_id in baby_indices:
        if baby_id in baby_states and baby_states[baby_id] is not None:
            states = baby_states[baby_id]
            counts = np.zeros(n_states)
            
            for s in range(n_states):
                counts[s] = np.sum(states == s)
            
            proportions = counts / len(states)
            state_usage[baby_id] = proportions
    
    # Create a heatmap of state distributions
    valid_babies = list(state_usage.keys())
    
    if valid_babies:
        usage_matrix = np.array([state_usage[baby_id] for baby_id in valid_babies])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(usage_matrix, 
                   cmap='viridis', 
                   xticklabels=[f'State {s}' for s in range(n_states)],
                   yticklabels=valid_babies,
                   cbar_kws={'label': 'Proportion'})
        plt.title('State Usage Distribution Across Babies')
        plt.xlabel('States')
        plt.ylabel('Baby ID')
        plt.tight_layout()
        
        # Save heatmap
        plt.savefig(f"{output_dir}/state_usage_heatmap.png", dpi=300)
        plt.close()
        
        plt.figure(figsize=(10, 6))
        overall_usage = usage_matrix.mean(axis=0)
        
        plt.bar(range(n_states), overall_usage, color='teal')
        plt.xlabel('State')
        plt.ylabel('Average Proportion')
        plt.title('Overall State Usage Across All Babies')
        plt.xticks(range(n_states), [f'State {s}' for s in range(n_states)])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, v in enumerate(overall_usage):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/overall_state_usage.png", dpi=300)
        plt.close()
        
        print(f"State usage plots saved to {output_dir}/")
    
    return state_usage

def plot_log_likelihood(log_probs, output_dir="output/training_plots"):
    """
    Plot the log likelihood progression during training and save to output folder.
    
    Args:
        log_probs (array): Log probabilities from training
        output_dir (str): Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(log_probs, '-o')
    plt.title('Log Likelihood Progression')
    plt.xlabel('EM Iteration')
    plt.ylabel('Log Likelihood')
    plt.grid(True)
    plt.tight_layout()
    
    # Save to file
    plt.savefig(f"{output_dir}/log_likelihood.png", dpi=300)
    plt.close()
    
    # Save the raw data
    np.savetxt(f"{output_dir}/log_likelihood_values.csv", log_probs, delimiter=",", 
               header="log_likelihood", comments='')
    
    print(f"Log likelihood plot saved to {output_dir}/log_likelihood.png")
    print(f"Log likelihood values saved to {output_dir}/log_likelihood_values.csv")

def arhmm_analysis(baby_pca_results, changepoint_average=None, output_dir="output", n_states=4):
    """
    Main AR-HMM analysis pipeline with output saved to files.
    
    Args:
        baby_pca_results (dict): PCA results for each baby
        changepoint_average (dict, optional): Changepoints for each baby
        output_dir (str): Base directory for saving outputs
        n_states (int): Number of discrete states to use in the model
    
    Returns:
        tuple: Inferred baby states and model information
    """
    output_dir = f"{output_dir}_{n_states}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting AR-HMM analysis with {n_states} states...")
    print(f"Output will be saved to: {output_dir}")
    
    # Prep data
    all_data_sequences_jax, baby_indices, obs_dim = prepare_data(baby_pca_results)
    print(f"Data prepared: {len(all_data_sequences_jax)} sequences, observation dimension: {obs_dim}")
    
    # AR-HMM parameters
    arhmm_params = {
        "n_states": n_states,   # Number of discrete states
        "ar_lags": 1,           # AR model lag order
        "max_em_iters": 20,     # Maximum EM iterations
        "em_tol": 1e-4,         # Convergence tolerance
        "seed": 7              # Random seed
    }
    
    model, params, param_props = create_arhmm_model(arhmm_params, obs_dim)
    print(f"Model created with {arhmm_params['n_states']} states and {arhmm_params['ar_lags']} lag(s)")
    
    # Train model
    trained_params, log_probs, sequence_boundaries = train_model_with_dynamax(
        model, params, param_props, all_data_sequences_jax, arhmm_params)
    print(f"Model trained for {len(log_probs)} iterations")
    
    plot_log_likelihood(log_probs, os.path.join(output_dir, "training_plots"))
    
    # Infer states and calculate log-likelihoods
    baby_states, log_likelihoods, norm_log_likelihoods = infer_states(
        model, trained_params, all_data_sequences_jax, baby_indices, sequence_boundaries)
    print(f"States inferred for {len(baby_states)} babies")
    
    # Visualizations
    state_plots_dir = os.path.join(output_dir, "state_plots")
    plot_stacked_states(baby_states, baby_indices, state_plots_dir)
    state_usage = plot_state_usage(baby_states, baby_indices, arhmm_params["n_states"], state_plots_dir)
    
    # Save model
    models_dir = os.path.join(output_dir, "arhmm_models")
    model_path, params_path = save_models(model, trained_params, models_dir)
    print(f"Models saved to {models_dir}")
    
    # Save state to CSV files
    states_dir = os.path.join(output_dir, "state_sequences")
    os.makedirs(states_dir, exist_ok=True)
    for baby_id, states in baby_states.items():
        if states is not None:
            np.savetxt(f"{states_dir}/baby_{baby_id}_states.csv", states, 
                       delimiter=",", fmt="%d", header="state")
    print(f"State sequences saved to {states_dir}/")
    
    results = {
        "baby_states": baby_states,
        "model": model,
        "params": trained_params,
        "log_probs": log_probs,
        "state_usage": state_usage,
        "arhmm_params": arhmm_params,
        "sequence_boundaries": sequence_boundaries,
    }
    
    print(f"Analysis complete. All outputs saved to {output_dir}/")
    return results