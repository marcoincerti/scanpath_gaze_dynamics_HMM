import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import multiprocessing
import logging

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam
from pyro import poutine
from pyro.distributions import constraints

def prepare_data(trials):
    """
    Prepara i dati per l'addestramento dell'HMM.

    Args:
        trials (list): Lista di dizionari di prove, ciascuno contenente 'Fixations'.

    Returns:
        data (torch.Tensor): Tensor dei dati con padding di forma (N, T_max, D).
        lengths (torch.Tensor): Lunghezze di ciascuna sequenza.
    """
    sequences = []
    lengths = []
    for trial in trials:
        fixations = trial['Fixations']
        lengths.append(len(fixations))
        sequences.append(fixations)
    
    # Padding delle sequenze alla stessa lunghezza
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
    Modello Hidden Markov con dati osservati e stati nascosti.

    Args:
        data (torch.Tensor): Tensor dei dati osservati di forma (N, T_max, D).
        lengths (torch.Tensor): Lunghezze di ciascuna sequenza.
        K (int): Numero di stati nascosti.

    Returns:
        None
    """
    T_max = data.size(1)  # Lunghezza massima delle sequenze
    N = data.size(0)      # Numero di sequenze
    D = data.size(2)      # Dimensione dei dati (es. FixX e FixY)

    # Parametri globali
    probs_z0 = pyro.param('probs_z0', torch.ones(K) / K, constraint=constraints.simplex)
    probs_z = pyro.param('probs_z', torch.ones(K, K) / K, constraint=constraints.simplex)
    mus = pyro.param('mus', torch.randn(K, D))
    sigmas = pyro.param('sigmas', torch.ones(K, D), constraint=constraints.positive)

    with pyro.plate('sequences', N):
        z_prev = None
        for t in pyro.markov(range(T_max)):
            # Maschera ogni sequenza basata sulla sua lunghezza
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
    Funzione guida per SVI. Vuota poiché utilizziamo l'enumerazione.

    Args:
        data (torch.Tensor): Tensor dei dati osservati.
        lengths (torch.Tensor): Lunghezze di ciascuna sequenza.
        K (int): Numero di stati nascosti.

    Returns:
        None
    """
    pass  # Guida vuota per l'enumerazione delle variabili discrete

def train_hmm_for_k(data, lengths, K, init, num_iters):
    """
    Addestra l'HMM per un dato numero di stati nascosti K.

    Args:
        data (torch.Tensor): Tensor dei dati osservati.
        lengths (torch.Tensor): Lunghezze di ciascuna sequenza.
        K (int): Numero di stati nascosti.
        init (int): Indice dell'inizializzazione.
        num_iters (int): Numero di iterazioni.

    Returns:
        final_loss (float): Valore finale della loss.
        detached_params (dict): Parametri addestrati (staccati dal grafo computazionale).
    """
    pyro.clear_param_store()
    optim = Adam({"lr": 0.01})
    elbo = TraceEnum_ELBO()
    svi = SVI(model, guide, optim, loss=elbo)

    for i in range(num_iters):
        loss = svi.step(data, lengths, K)
        if i % 10 == 0:
            print(f"K={K}, Inizializzazione {init+1}, Iterazione {i}, Loss: {loss}")

    # Stacca i parametri per consentire il multiprocessing
    detached_params = {k: v.detach().clone() for k, v in pyro.get_param_store().items()}
    return loss, detached_params

class HMMModel:
    """
    Classe per memorizzare i migliori parametri del modello HMM e le predizioni.
    """
    def __init__(self, best_loss, best_params, best_K):
        self.best_loss = best_loss
        self.best_params = best_params
        self.best_K = best_K

    def predict(self, X):
        """
        Predice gli stati nascosti (ROI) per nuovi dati X.

        Args:
            X (numpy.ndarray o torch.Tensor): Dati osservati di forma (T, D).

        Returns:
            numpy.ndarray: Stati nascosti predetti di forma (T,).
        """
        # Converte X in torch.Tensor se necessario
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float)
        elif not isinstance(X, torch.Tensor):
            raise ValueError("L'input X deve essere un array numpy o un tensor torch.")

        # Ottiene i parametri
        probs_z0 = self.best_params['probs_z0']  # Forma: (K,)
        probs_z = self.best_params['probs_z']    # Forma: (K, K)
        mus = self.best_params['mus']            # Forma: (K, D)
        sigmas = self.best_params['sigmas']      # Forma: (K, D)

        # Implementa l'algoritmo di Viterbi
        K = probs_z0.shape[0]
        T = X.shape[0]
        D = X.shape[1]

        # Inizializza le matrici delle probabilità logaritmiche
        log_delta = torch.zeros(T, K)
        psi = torch.zeros(T, K, dtype=torch.long)

        # Precalcola le probabilità logaritmiche
        log_probs_z0 = torch.log(probs_z0 + 1e-8)
        log_probs_z = torch.log(probs_z + 1e-8)

        # Calcola le probabilità logaritmiche di emissione per tutti i tempi e stati
        log_probs_emission = torch.zeros(T, K)
        for k in range(K):
            emission_dist = dist.Normal(mus[k], sigmas[k])
            log_probs_emission[:, k] = emission_dist.log_prob(X).sum(dim=1)  # Somma su D

        # Inizializzazione
        log_delta[0] = log_probs_z0 + log_probs_emission[0]

        # Ricorsione
        for t in range(1, T):
            for k in range(K):
                log_transition = log_delta[t-1] + log_probs_z[:, k]
                max_log_transition, argmax_k = torch.max(log_transition, dim=0)
                log_delta[t, k] = max_log_transition + log_probs_emission[t, k]
                psi[t, k] = argmax_k

        # Terminazione
        path = torch.zeros(T, dtype=torch.long)
        max_log_prob, last_state = torch.max(log_delta[T-1], dim=0)
        path[T-1] = last_state

        # Backtracking del percorso
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]

        # Converte in array numpy
        return path.numpy()
