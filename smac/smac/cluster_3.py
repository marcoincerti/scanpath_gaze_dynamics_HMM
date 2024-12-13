import numpy as np
from scipy.special import psi
from sklearn.cluster import KMeans

def compute_sufficient_statistics(hmm_model):
    """
    Compute sufficient statistics from an HMM model.
    This includes transition counts and expected emission statistics.
    """
    A = hmm_model.alpha / hmm_model.alpha.sum(axis=1, keepdims=True)  # Transition matrix
    mus = hmm_model.m  # Emission means
    covs = [np.linalg.inv(hmm_model.nu[k] * hmm_model.W[k]) for k in range(hmm_model.K)]  # Covariance matrices
    
    return {
        'transitions': A,
        'means': mus,
        'covariances': covs,
        'initial_probs': hmm_model.alpha.sum(axis=1)  # Prior probabilities
    }

def compute_kl_divergence(hmm1, hmm2):
    """ww
    Compute the KL divergence between two HMMs based on their parameters.
    """
    kl_div = 0
    # KL divergence for transition matrices
    kl_div += np.sum(hmm1['transitions'] * np.log(hmm1['transitions'] / hmm2['transitions']))
    # KL divergence for emissions (Gaussian)
    for mu1, cov1, mu2, cov2 in zip(hmm1['means'], hmm1['covariances'], hmm2['means'], hmm2['covariances']):
        kl_div += np.trace(np.linalg.solve(cov2, cov1)) + \
                  (mu2 - mu1).T @ np.linalg.solve(cov2, (mu2 - mu1)) - len(mu1) + \
                  np.log(np.linalg.det(cov2) / np.linalg.det(cov1))
    return kl_div

def vhem_cluster_models(hmm_models, num_clusters, max_iter=100, tol=1e-4):
    """
    Cluster HMM models using the Variational Hierarchical Expectation Maximization (VHEM) approach.
    """
    # Step 1: Initialize clusters randomly or with k-means on transition matrices
    initial_centroids = KMeans(n_clusters=num_clusters, random_state=0).fit(
        [hmm['transitions'].flatten() for hmm in hmm_models]
    ).cluster_centers_
    cluster_hmms = [
        {'transitions': centroid.reshape(-1, centroid.shape[0]),
         'means': [np.mean(hmm['means'], axis=0) for hmm in hmm_models],  # Initialize as mean of all
         'covariances': [np.mean([hmm['covariances'][k] for hmm in hmm_models], axis=0) for k in range(len(hmm_models[0]['means']))],
         'initial_probs': np.mean([hmm['initial_probs'] for hmm in hmm_models], axis=0)}
        for centroid in initial_centroids
    ]
    
    # Step 2: Iterative optimization (VHEM)
    for iteration in range(max_iter):
        # E-Step: Compute soft assignments
        responsibilities = np.zeros((len(hmm_models), num_clusters))
        for i, hmm in enumerate(hmm_models):
            for k, cluster_hmm in enumerate(cluster_hmms):
                responsibilities[i, k] = -compute_kl_divergence(hmm, cluster_hmm)
        responsibilities = np.exp(responsibilities - responsibilities.max(axis=1, keepdims=True))
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        
        # M-Step: Update cluster HMMs
        new_cluster_hmms = []
        for k in range(num_clusters):
            weight_sum = responsibilities[:, k].sum()
            weighted_sums = {
                'transitions': np.sum([r * hmm['transitions'] for r, hmm in zip(responsibilities[:, k], hmm_models)], axis=0) / weight_sum,
                'means': np.sum([r * hmm['means'] for r, hmm in zip(responsibilities[:, k], hmm_models)], axis=0) / weight_sum,
                'covariances': np.sum([r * np.array(hmm['covariances']) for r, hmm in zip(responsibilities[:, k], hmm_models)], axis=0) / weight_sum,
                'initial_probs': np.sum([r * hmm['initial_probs'] for r, hmm in zip(responsibilities[:, k], hmm_models)], axis=0) / weight_sum,
            }
            new_cluster_hmms.append(weighted_sums)
        
        # Check for convergence
        if all(np.linalg.norm(new_hmm['transitions'] - old_hmm['transitions']) < tol
               for new_hmm, old_hmm in zip(new_cluster_hmms, cluster_hmms)):
            break
        cluster_hmms = new_cluster_hmms
    
    return responsibilities.argmax(axis=1), cluster_hmms