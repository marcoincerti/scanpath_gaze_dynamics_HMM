import logging
from hmmlearn import vhmm
from sklearn.mixture import BayesianGaussianMixture
import pandas as pd
import torch

def predict_rois(subject, data, best_model_tuple):
    """Predicts Regions of Interest (ROIs) for a given subject using the best model."""
    model, scaler = best_model_tuple
    subject_data = data[data['SubjectID'] == subject]

    if subject_data.empty:
        logging.warning(f"No data available for Subject {subject}.")
        return None

    # Extract and scale the fixation points
    X = subject_data[['FixX', 'FixY']].values
    if X.size == 0:
        logging.warning(f"No fixation points available for Subject {subject}.")
        return None

    X_scaled = scaler.transform(X)

    # Predict hidden states based on the model type
    if isinstance(model, vhmm.VariationalGaussianHMM):
        lengths = subject_data.groupby('TrialID').size().tolist()
        hidden_states = model.predict(X_scaled, lengths)
    elif isinstance(model, BayesianGaussianMixture):
        hidden_states = model.predict(X_scaled)
    elif isinstance(model, dict) and "initial_probs" in model:
        # Use the VBEM-HMM model for prediction
        X_tensor = torch.tensor(X_scaled, dtype=torch.float)

        # Extract learned parameters from the VBEM-HMM model
        initial_probs = torch.tensor(model["initial_probs"])
        transition_matrix = torch.tensor(model["transition_matrix"])
        emission_means = torch.tensor(model["emission_means"])
        emission_scales = torch.tensor(model["emission_scales"])

        # Perform Viterbi decoding to predict the most likely hidden state sequence
        T = X_tensor.shape[0]
        log_probs = torch.zeros((T, len(initial_probs)))

        # Initialize with log probabilities of the initial states
        log_probs[0] = torch.log(initial_probs) + torch.distributions.Normal(
            emission_means, emission_scales).log_prob(X_tensor[0]).sum(dim=-1)

        # Dynamic programming to compute log probabilities
        for t in range(1, T):
            transition_log_probs = torch.log(transition_matrix) + log_probs[t - 1].unsqueeze(1)
            log_probs[t] = torch.max(transition_log_probs, dim=0).values + torch.distributions.Normal(
                emission_means, emission_scales).log_prob(X_tensor[t]).sum(dim=-1)

        # Backtrace to find the most likely state sequence
        hidden_states = torch.argmax(log_probs, dim=1).numpy()
    else:
        logging.error(f"Unknown model type: {type(model)}")
        return None

    # Add the hidden states to the subject data
    subject_data = subject_data.copy()
    subject_data['hidden_state'] = hidden_states

    logging.info(f"Predicted hidden states for Subject {subject}.")

    return subject_data