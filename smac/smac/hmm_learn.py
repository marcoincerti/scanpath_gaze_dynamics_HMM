import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import multiprocessing

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam
from pyro import poutine
from pyro.distributions import constraints

def prepare_data(trials):
    """
    Prepares the data for HMM training.

    Args:
        trials (list): A list of trial dictionaries, each containing 'Fixations'.

    Returns:
        data (torch.Tensor): Padded data tensor of shape (N, T_max, D).
        lengths (torch.Tensor): Lengths of each sequence.
    """
    sequences = []
    lengths = []
    for trial in trials:
        fixations = trial['Fixations']
        lengths.append(len(fixations))
        sequences.append(fixations)
    
    # Pad sequences to the same length
    max_length = max(lengths)
    padded_sequences = []
    for seq in sequences:
        pad_size = max_length - len(seq)
        if pad_size > 0:
            padding = np.zeros((pad_size, seq.shape[1]))
            seq_padded = np.vstack([seq, padding])
        else:
            seq_padded = seq
        padded_sequences.append(seq_padded)
    
    data = np.stack(padded_sequences)
    data = torch.tensor(data, dtype=torch.float)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return data, lengths

@config_enumerate
def model(data, lengths, K):
    """
    Hidden Markov Model with observed data and hidden states.

    Args:
        data (torch.Tensor): Observed data tensor of shape (N, T_max, D).
        lengths (torch.Tensor): Lengths of each sequence.
        K (int): Number of hidden states.

    Returns:
        None
    """
    T_max = data.size(1)  # Maximum sequence length
    N = data.size(0)      # Number of sequences
    D = data.size(2)      # Data dimensionality (e.g., FixX and FixY)

    # Global parameters
    probs_z0 = pyro.param('probs_z0', torch.ones(K) / K, constraint=constraints.simplex)
    probs_z = pyro.param('probs_z', torch.ones(K, K) / K, constraint=constraints.simplex)
    mus = pyro.param('mus', torch.randn(K, D))
    sigmas = pyro.param('sigmas', torch.ones(K, D), constraint=constraints.positive)

    with pyro.plate('sequences', N):
        z_prev = None
        for t in pyro.markov(range(T_max)):
            # Mask each sequence based on its length
            mask_t = (t < lengths).bool()
            with poutine.mask(mask=mask_t):
                if t == 0:
                    z_t = pyro.sample(f'z_{t}', dist.Categorical(probs_z0))
                else:
                    z_t = pyro.sample(f'z_{t}', dist.Categorical(probs_z[z_prev]))
                pyro.sample(f'obs_{t}', dist.Normal(mus[z_t], sigmas[z_t]).to_event(1), obs=data[:, t, :])
                z_prev = z_t

def guide(data, lengths, K):
    """
    Guide function for SVI. Empty since we're using enumeration.

    Args:
        data (torch.Tensor): Observed data tensor.
        lengths (torch.Tensor): Lengths of each sequence.
        K (int): Number of hidden states.

    Returns:
        None
    """
    pass  # Empty guide for enumeration of discrete variables

def train_hmm_for_k(data, lengths, K, init, num_iters):
    """
    Trains the HMM for a given number of hidden states K.

    Args:
        data (torch.Tensor): Observed data tensor.
        lengths (torch.Tensor): Lengths of each sequence.
        K (int): Number of hidden states.
        init (int): Initialization index.
        num_iters (int): Number of iterations.

    Returns:
        final_loss (float): Final loss value.
        detached_params (dict): Trained parameters (detached).
    """
    pyro.clear_param_store()
    optim = Adam({"lr": 0.01})
    elbo = TraceEnum_ELBO()
    svi = SVI(model, guide, optim, loss=elbo)

    for i in range(num_iters):
        loss = svi.step(data, lengths, K)
        if i % 10 == 0:
            print(f"K={K}, Initialization {init+1}, Iteration {i}, Loss: {loss}")

    # Detach parameters to allow multiprocessing
    detached_params = {k: v.detach().clone() for k, v in pyro.get_param_store().items()}
    return loss, detached_params

class HMMModel:
    """
    Class to store the best HMM model parameters and predictions.
    """
    def __init__(self, best_loss, best_params, best_K):
        self.best_loss = best_loss
        self.best_params = best_params
        self.best_K = best_K

    def predict(self, X):
        """
        Predicts the hidden states (ROIs) for new data X.

        Args:
            X (numpy.ndarray or torch.Tensor): Observed data of shape (T, D).

        Returns:
            numpy.ndarray: Predicted hidden states of shape (T,).
        """
        # Convert X to torch.Tensor if necessary
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float)
        elif not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a numpy array or torch tensor.")

        # Get the parameters
        probs_z0 = self.best_params['probs_z0']  # Shape: (K,)
        probs_z = self.best_params['probs_z']    # Shape: (K, K)
        mus = self.best_params['mus']            # Shape: (K, D)
        sigmas = self.best_params['sigmas']      # Shape: (K, D)

        # Implement the Viterbi algorithm
        K = probs_z0.shape[0]
        T = X.shape[0]
        D = X.shape[1]

        # Initialize the log probabilities matrices
        log_delta = torch.zeros(T, K)
        psi = torch.zeros(T, K, dtype=torch.long)

        # Precompute log probabilities
        log_probs_z0 = torch.log(probs_z0 + 1e-8)
        log_probs_z = torch.log(probs_z + 1e-8)

        # Compute emission log probabilities for all time steps and states
        log_probs_emission = torch.zeros(T, K)
        for k in range(K):
            emission_dist = dist.Normal(mus[k], sigmas[k])
            log_probs_emission[:, k] = emission_dist.log_prob(X).sum(dim=1)  # Sum over D

        # Initialization
        log_delta[0] = log_probs_z0 + log_probs_emission[0]

        # Recursion
        for t in range(1, T):
            for k in range(K):
                log_transition = log_delta[t-1] + log_probs_z[:, k]
                max_log_transition, argmax_k = torch.max(log_transition, dim=0)
                log_delta[t, k] = max_log_transition + log_probs_emission[t, k]
                psi[t, k] = argmax_k

        # Termination
        path = torch.zeros(T, dtype=torch.long)
        max_log_prob, last_state = torch.max(log_delta[T-1], dim=0)
        path[T-1] = last_state

        # Path backtracking
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]

        # Convert to numpy array
        return path.numpy()

def fit_hmm_pyro(data, lengths, roi_values, num_iters=500, num_init=5):
    """
    Fits an HMM to the data using Pyro for different values of K (number of hidden states).

    Args:
        data (torch.Tensor): Observed data tensor.
        lengths (torch.Tensor): Lengths of each sequence.
        roi_values (list): List of K values to try.
        num_iters (int): Number of iterations per initialization.
        num_init (int): Number of initializations per K.

    Returns:
        HMMModel: Best HMM model found.
    """
    best_loss = float('inf')
    best_params = None
    best_K = None

    for K in roi_values:
        print(f"Trying model with K={K} hidden states...")

        with multiprocessing.Pool(processes=num_init) as pool:
            results = pool.starmap(
                train_hmm_for_k,
                [(data, lengths, K, init, num_iters) for init in range(num_init)]
            )

        for final_loss, params in results:
            if final_loss < best_loss:
                best_loss = final_loss
                best_params = params
                best_K = K

    # Restore best parameters into Pyro's param store
    pyro.get_param_store().clear()
    for k, v in best_params.items():
        pyro.get_param_store()[k] = v.clone()

    return HMMModel(best_loss, best_params, best_K)