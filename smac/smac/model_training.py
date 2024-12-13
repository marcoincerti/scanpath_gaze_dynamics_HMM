import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import BayesianGaussianMixture
from hmmlearn import hmm, vhmm
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.optim import Adam
import torch
from torch.distributions import constraints

def calculate_bic(log_likelihood, n_params, n_samples):
    """Calculates the Bayesian Information Criterion (BIC)."""
    return -2 * log_likelihood + n_params * np.log(n_samples)

def train_vhmm(X_scaled, lengths, n_components, max_iter=1000):
    """Trains a Variational Gaussian HMM and selects the best one based on BIC."""
    try:
        # Initialize the VariationalGaussianHMM model
        model = vhmm.VariationalGaussianHMM(
            n_components=n_components,
            covariance_type='full',
            n_iter=max_iter,
            tol=1e-2,
            random_state=42
        )
        # Fit the model to the scaled data
        model.fit(X_scaled, lengths)
        log_likelihood = model.score(X_scaled, lengths)
        n_params = (
            n_components * X_scaled.shape[1]  # Gaussian parameters
            + n_components ** 2  # Transition matrix
            - 1  # One parameter is redundant
        )
        bic = calculate_bic(log_likelihood, n_params, X_scaled.shape[0])

        logging.info(
            f"VHMM - Components: {n_components}, Log Likelihood: {log_likelihood:.2f}, BIC: {bic:.2f}"
        )

        return model, bic

    except Exception as e:
        logging.error(f"Error training VariationalGaussianHMM with {n_components} components: {e}")
        return None, np.inf

def train_bayesian_gmm(X_scaled, n_components, max_iter=1000):
    """Trains a Bayesian Gaussian Mixture Model and returns the model and BIC."""
    best_model = None
    lowest_bic = np.inf

    model = BayesianGaussianMixture(
        n_components=n_components,
        covariance_type='full',
        max_iter=max_iter,
        weight_concentration_prior_type='dirichlet_process',
        random_state=42
    )

    try:
        model.fit(X_scaled)
        log_likelihood = model.score(X_scaled) * X_scaled.shape[0]
        n_params = n_components * (
            X_scaled.shape[1] + X_scaled.shape[1] * (X_scaled.shape[1] + 1) / 2
        )
        bic = calculate_bic(log_likelihood, n_params, X_scaled.shape[0])

        logging.info(
            f"Bayesian GMM - Components: {n_components}, Log Likelihood: {log_likelihood:.2f}, BIC: {bic:.2f}"
        )

        if bic < lowest_bic:
            lowest_bic = bic
            best_model = model

    except Exception as e:
        logging.error(f"Error training BayesianGaussianMixture with {n_components} components: {e}")

    return best_model, lowest_bic

def train_vbem_hmm(X_scaled, n_components, max_iter=1000):
    """Trains a Variational Bayesian HMM using Pyro."""
    pyro.clear_param_store()  # Clear previous parameters

    # Convert X_scaled to a PyTorch tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float)

    # Define the HMM model using Pyro
    def model(X):
        # Initial state distribution
        initial_probs = pyro.sample("initial_probs", dist.Dirichlet(torch.ones(n_components)))
        # Transition matrix
        transition_matrix = pyro.sample("transition_matrix", dist.Dirichlet(torch.ones(n_components, n_components), independent=True))
        
        # Emission means and covariances
        with pyro.plate("states", n_components):
            emission_means = pyro.sample("emission_means", dist.Normal(torch.zeros(X.shape[1]), torch.ones(X.shape[1])).to_event(1))
            emission_scales = pyro.sample("emission_scales", dist.LogNormal(torch.zeros(X.shape[1]), torch.ones(X.shape[1])).to_event(1))
        
        # Define emissions for each observation
        for t in range(X.shape[0]):
            # Sample state
            state = pyro.sample(f"state_{t}", dist.Categorical(initial_probs if t == 0 else transition_matrix[state]))
            # Sample emission
            pyro.sample(f"obs_{t}", dist.Normal(emission_means[state], emission_scales[state]).to_event(1), obs=X[t])

    # Define the guide (variational distribution)
    def guide(X):
        # Variational parameters for initial state probabilities
        initial_probs_q = pyro.param("initial_probs_q", torch.ones(n_components), constraint=constraints.simplex)
        # Variational parameters for transition matrix
        transition_matrix_q = pyro.param("transition_matrix_q", torch.ones(n_components, n_components), constraint=constraints.simplex)
        # Variational parameters for emission means and scales
        with pyro.plate("states", n_components):
            emission_means_q = pyro.param("emission_means_q", torch.zeros(X.shape[1]))
            emission_scales_q = pyro.param("emission_scales_q", torch.ones(X.shape[1]), constraint=constraints.positive)

        # Sample variational distributions
        pyro.sample("initial_probs", dist.Dirichlet(initial_probs_q))
        pyro.sample("transition_matrix", dist.Dirichlet(transition_matrix_q).to_event(1))  # Use .to_event(1) instead of independent=True
        with pyro.plate("states", n_components):
            pyro.sample("emission_means", dist.Normal(emission_means_q, torch.ones(X.shape[1])).to_event(1))
            pyro.sample("emission_scales", dist.LogNormal(emission_scales_q, torch.ones(X.shape[1])).to_event(1))

    # Set up the optimizer and inference algorithm
    optimizer = Adam({"lr": 0.01})
    svi = SVI(model, guide, optimizer, loss=TraceMeanField_ELBO())

    # Perform variational inference
    for step in range(max_iter):
        loss = svi.step(X_tensor)
        if step % 100 == 0:
            logging.info(f"Step {step}: Loss = {loss}")

    # Collect learned parameters
    initial_probs_learned = pyro.param("initial_probs_q").detach().numpy()
    transition_matrix_learned = pyro.param("transition_matrix_q").detach().numpy()
    emission_means_learned = pyro.param("emission_means_q").detach().numpy()
    emission_scales_learned = pyro.param("emission_scales_q").detach().numpy()

    # Return learned parameters as the "model"
    learned_model = {
        "initial_probs": initial_probs_learned,
        "transition_matrix": transition_matrix_learned,
        "emission_means": emission_means_learned,
        "emission_scales": emission_scales_learned
    }

    return learned_model

def train_hmm(subject, data, num_states_list, use_vbem=False, use_bayesian_gmm=True, use_gmm_hmm=False, n_inits=10, max_iter=1000):
    """Trains HMM models using different approaches and selects the best model based on BIC."""
    logging.info(f"Training models for Subject {subject}.")

    subject_data = data[data['SubjectID'] == subject]
    sequences = []
    lengths = []

    for trial in subject_data['TrialID'].unique():
        trial_data = subject_data[subject_data['TrialID'] == trial]
        sequence = trial_data[['FixX', 'FixY']].values
        sequences.append(sequence)
        lengths.append(len(sequence))

    if not sequences:
        logging.warning(f"No data available for Subject {subject}.")
        return None

    # Prepare the data
    X = np.concatenate(sequences)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Variables to track the best model
    best_model = None
    lowest_bic = np.inf
    model_type = None

    # Iterate over the number of states to train different models
    for n_components in num_states_list:
        # Train GMM-HMM models if enabled
        if use_gmm_hmm:
            vhmm_model, vhmm_bic = train_vhmm(X_scaled, lengths, n_components, max_iter=max_iter)
            if vhmm_bic < lowest_bic:
                lowest_bic = vhmm_bic
                best_model = vhmm_model
                model_type = 'VHMM'

        # Train VBEM HMM models if enabled
        if use_vbem:
            vbem_hmm_model = train_vbem_hmm(X_scaled, n_components, max_iter=max_iter)
            vbem_bic = np.inf  # PlacxeCholder for BIC calculation
            if vbem_bic < lowest_bic:
                lowest_bic = vbem_bic
                best_model = vbem_hmm_model
                model_type = 'VBEM-HMM'

        # Train Bayesian GMM models if enabled
        if use_bayesian_gmm:
            bgmm_model, bgmm_bic = train_bayesian_gmm(X_scaled, n_components, max_iter=max_iter)
            if bgmm_bic < lowest_bic:
                lowest_bic = bgmm_bic
                best_model = bgmm_model
                model_type = 'BayesianGMM'

    # Log and return the best model
    if best_model:
        logging.info(f"Best model for Subject {subject}: {model_type} with BIC {lowest_bic:.2f}")
        return best_model, scaler
    else:
        logging.error(f"No valid model found for Subject {subject}.")
        return None