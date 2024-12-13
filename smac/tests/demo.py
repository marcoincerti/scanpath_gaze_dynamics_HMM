import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam
from pyro import poutine
from pyro.distributions import constraints
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar

def read_fixation_data(file_path):
    data = pd.read_excel(file_path)
    subjects = data['SubjectID'].unique()
    subject_data = {}

    for subject in subjects:
        subject_trials = data[data['SubjectID'] == subject]
        trials = subject_trials['TrialID'].unique()
        trial_data = []
        for trial in trials:
            trial_fixations = subject_trials[subject_trials['TrialID'] == trial][['FixX', 'FixY']].values
            trial_data.append(trial_fixations)
        subject_data[subject] = trial_data

    return subject_data

def prepare_data(trial_data):
    sequences = []
    lengths = []
    all_data = []

    for trial in trial_data:
        sequences.append(torch.tensor(trial, dtype=torch.float32))
        lengths.append(len(trial))
        all_data.append(trial)

    # Normalize data
    all_data = np.vstack(all_data)
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)
    sequences = [(seq - torch.tensor(mean)) / torch.tensor(std) for seq in sequences]

    lengths = torch.tensor(lengths, dtype=torch.long)
    max_length = max(lengths)
    obs_dim = sequences[0].shape[1]
    padded_sequences = torch.zeros(len(sequences), max_length, obs_dim)

    for i, seq in enumerate(sequences):
        padded_sequences[i, :lengths[i]] = seq

    return padded_sequences, lengths

@config_enumerate
def model(sequences, lengths, num_states):
    batch_size = len(lengths)
    max_length = sequences.size(1)
    obs_dim = sequences.size(2)

    # Global parameters
    probs_x = pyro.sample("probs_x", dist.Dirichlet(0.5 * torch.ones(num_states)))
    
    # Use .to_event(1) to handle the transition matrix
    probs_z = pyro.sample("probs_z", dist.Dirichlet(0.5 * torch.ones(num_states, num_states)).to_event(1))

    # Emission parameters
    locs = pyro.sample("locs", dist.Normal(torch.zeros(num_states, obs_dim), 10.0 * torch.ones(num_states, obs_dim)).to_event(2))
    scales = pyro.sample("scales", dist.Normal(torch.ones(num_states, obs_dim), torch.ones(num_states, obs_dim)).to_event(2))

    with pyro.plate("sequences", batch_size):
        lengths = lengths.long()
        x = sequences  # Shape: [batch_size, max_length, obs_dim]
        x_mask = torch.arange(max_length).unsqueeze(0) < lengths.unsqueeze(1)
        x_mask = x_mask.to(torch.bool)

        z_prev = None
        for t in pyro.markov(range(max_length)):
            with pyro.poutine.mask(mask=x_mask[:, t]):
                if z_prev is None:
                    probs_t = probs_x
                else:
                    probs_t = probs_z[z_prev]  # Direct indexing for transitions
                z_t = pyro.sample(f"z_{t}", dist.Categorical(probs_t), infer={"enumerate": "parallel"})
                pyro.sample(f"x_{t}", dist.Normal(locs[z_t], scales[z_t]).to_event(1), obs=x[:, t])
                z_prev = z_t

def guide(sequences, lengths, num_states):
    batch_size = len(lengths)
    max_length = sequences.size(1)
    obs_dim = sequences.size(2)

    # Variational parameters for the initial state probabilities
    probs_x_posterior = pyro.param("probs_x_posterior", torch.ones(num_states), constraint=constraints.simplex)
    pyro.sample("probs_x", dist.Dirichlet(probs_x_posterior))

    # Variational parameters for the transition probabilities
    probs_z_posterior = pyro.param("probs_z_posterior", torch.ones(num_states, num_states), constraint=constraints.simplex)
    
    # Apply .to_event(1) to handle the batch dimension
    pyro.sample("probs_z", dist.Dirichlet(probs_z_posterior).to_event(1))

    # Variational parameters for emission parameters
    locs_posterior = pyro.param("locs_posterior", torch.zeros(num_states, obs_dim))
    scales_posterior = pyro.param("scales_posterior", torch.ones(num_states, obs_dim), constraint=constraints.positive)
    pyro.sample("locs", dist.Normal(locs_posterior, 0.1 * torch.ones(num_states, obs_dim)).to_event(2))
    pyro.sample("scales", dist.LogNormal(scales_posterior, 0.1 * torch.ones(num_states, obs_dim)).to_event(2))

def train_model(sequences, lengths, num_states, num_steps=500):
    pyro.clear_param_store()

    optimizer = Adam({"lr": 0.005})
    elbo = TraceEnum_ELBO(max_plate_nesting=1)
    svi = SVI(model, guide, optimizer, loss=elbo)

    losses = []
    # Add progress bar
    for step in tqdm(range(num_steps), desc=f"Training HMM with {num_states} states"):
        loss = svi.step(sequences, lengths, num_states)
        losses.append(loss)
        if step % 100 == 0:  # Log every 100 steps
            print(f"Step {step}: Loss = {loss:.4f}")

    return losses

def predict_hidden_states(sequences, lengths, num_states, params):
    pyro.clear_param_store()
    for name, value in params.items():
        pyro.param(name, value)

    probs_x = pyro.param("probs_x_posterior")
    probs_z = pyro.param("probs_z_posterior")
    locs = pyro.param("locs_posterior")
    scales = pyro.param("scales_posterior")

    batch_size = len(lengths)
    max_length = sequences.size(1)
    obs_dim = sequences.size(2)

    x = sequences
    x_mask = torch.arange(max_length).unsqueeze(0) < lengths.unsqueeze(1)
    x_mask = x_mask.to(torch.bool)

    z_map = []  # To store predicted hidden states
    emission_probs_list = []  # To store emission probabilities

    with torch.no_grad():
        z_prev = None
        for t in range(max_length):
            x_t = x[:, t]
            if z_prev is None:
                probs_t = probs_x.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, num_states]
            else:
                probs_t = probs_z[z_prev.squeeze(-1)]  # Shape: [batch_size, num_states]

            # Compute emission probabilities for each hidden state
            emission_probs = dist.Normal(locs, scales).log_prob(x_t.unsqueeze(1)).sum(-1).exp()  # Shape: [batch_size, num_states]

            # Store emission probabilities for debugging or analysis
            emission_probs_list.append(emission_probs.cpu().numpy())

            # Compute posterior over hidden states (probs_t * emission_probs)
            posterior = probs_t * emission_probs
            posterior = posterior / posterior.sum(-1, keepdim=True)

            # Most probable state at current time step
            z_t = posterior.argmax(-1, keepdim=True)  # Shape: [batch_size, 1]
            z_map.append(z_t)
            z_prev = z_t

    # Stack the predicted hidden states along the time axis
    z_map = torch.cat(z_map, dim=1)  # Shape: [batch_size, max_length]

    # Return both the hidden states and the emission probabilities
    return z_map, np.array(emission_probs_list)

if __name__ == "__main__":
    file_path = '/Users/marco.incerti/Desktop/uni-projects/INMCA_Incerti/smac/tests/demodata.xls'
    subject_data = read_fixation_data(file_path)

    num_states_list = [2, 3]
    subject_models = {}

    for subject, trials in subject_data.items():
        print(f"\nTraining HMM for Subject {subject}")

        best_loss = float('inf')
        best_num_states = None
        best_params = None

        for num_states in num_states_list:
            print(f"\nEvaluating HMM with {num_states} hidden states...")
            sequences, lengths = prepare_data(trials)
            losses = train_model(sequences, lengths, num_states, num_steps=1000)

            final_loss = losses[-1]
            print(f"Num States: {num_states}, Final Loss: {final_loss:.4f}")

            if final_loss < best_loss:
                best_loss = final_loss
                best_num_states = num_states
                best_params = {name: pyro.param(name).detach().clone() for name in pyro.get_param_store().keys()}
                print(f"Best model so far: {best_num_states} hidden states with loss {best_loss:.4f}")

        print(f"Selected {best_num_states} hidden states for Subject {subject}")

        # Save the best model parameters
        subject_models[subject] = (best_num_states, best_params)

        # Predict hidden states
        sequences, lengths = prepare_data(trials)
        z_map, emission_probs_array = predict_hidden_states(sequences, lengths, best_num_states, best_params)

        # Visualize data
        all_data = sequences.view(-1, sequences.size(2)).numpy()
        all_states = z_map.view(-1).numpy()

        plt.figure(figsize=(8, 6))
        plt.scatter(all_data[:, 0], all_data[:, 1], c=all_states, cmap='viridis', s=5)
        plt.xlabel('FixX')
        plt.ylabel('FixY')
        plt.title(f'Data Points Colored by Predicted Hidden States for Subject {subject}')
        plt.show()

        # Now you can analyze or visualize zones of interest based on 'z_map'
        print(f"Predicted hidden states for Subject {subject}:")
        print(z_map)