from sklearn.cluster import KMeans
import numpy as np
import logging

def extract_model_features(model, max_components=10):
    """
    Extracts and flattens the parameters of a BayesianGMM or HMM model,
    padding to a fixed number of components.

    Parameters:
        model: Trained model with attributes 'weights_', 'means_', and 'covariances_'.
        max_components (int): The maximum number of components to pad to.

    Returns:
        features (np.ndarray): Flattened and padded feature vector.
    """
    # Check if the model has the required attributes
    if not all(hasattr(model, attr) for attr in ['weights_', 'means_', 'covariances_']):
        raise ValueError("Model must have 'weights_', 'means_', and 'covariances_' attributes.")

    # Extract the number of components and feature dimension
    num_components = model.means_.shape[0]
    feature_dim = model.means_.shape[1]

    # Limit the number of components to max_components
    num_components = min(num_components, max_components)

    # Sort the components by their weights in descending order
    sorted_indices = np.argsort(-model.weights_)[:num_components]

    # Extract and sort weights, means, and covariances
    weights = model.weights_[sorted_indices]
    means = model.means_[sorted_indices]
    covariances = model.covariances_[sorted_indices]

    # Flatten means and covariances
    means_flat = means.flatten()
    covariances_flat = covariances.reshape(num_components, -1).flatten()

    # Pad weights, means, and covariances to ensure a fixed feature size
    weights_padded = np.pad(weights, (0, max_components - num_components), 'constant')
    means_padded = np.pad(means_flat, (0, max_components * feature_dim - means_flat.size), 'constant')
    cov_size = covariances_flat.size
    cov_expected_size = max_components * covariances_flat.size // num_components
    covariances_padded = np.pad(covariances_flat, (0, cov_expected_size - cov_size), 'constant')

    # Concatenate all features into a single vector
    features = np.concatenate([weights_padded, means_padded, covariances_padded])
    return features

def vhem_cluster_models(models, n_clusters=2, max_components=10):
    """
    Clusters the BayesianGMM or HMM models into groups using a KMeans-like approach.

    Parameters:
        models (list): List of trained models.
        n_clusters (int): Number of clusters to form.
        max_components (int): Maximum number of components to consider in each model.

    Returns:
        cluster_labels (np.ndarray): Cluster labels for each model.
        kmeans (KMeans): Trained KMeans clustering model.
    """
    # Extract features from each model
    feature_list = []
    for idx, model in enumerate(models):
        try:
            features = extract_model_features(model, max_components)
            feature_list.append(features)
        except ValueError as e:
            logging.warning(f"Skipping model at index {idx} due to error: {e}")
            continue

    if not feature_list:
        raise ValueError("No valid models provided for clustering.")

    feature_matrix = np.array(feature_list)
    
    # Normalize the feature matrix to have zero mean and unit variance
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)

    # Use KMeans to cluster the models based on their extracted features
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(feature_matrix_scaled)
    
    return cluster_labels, kmeans

import numpy as np
from sklearn.mixture import BayesianGaussianMixture

class VHEMClustering:
    def __init__(self, base_hmms, n_clusters):
        """
        Variational HEM clustering for HMMs.
        :param base_hmms: List of pre-trained BayesianGaussianMixture HMM models.
        :param n_clusters: Number of clusters to reduce to.
        """
        self.base_hmms = base_hmms
        self.n_clusters = n_clusters
        self.num_base_hmms = len(base_hmms)
        self.cluster_centers = [self._initialize_cluster_center() for _ in range(n_clusters)]
        self.assignments = np.zeros((self.num_base_hmms, n_clusters))  # Responsibility matrix

    def _initialize_cluster_center(self):
        """Randomly initialize a cluster center using a copy of a random base HMM."""
        random_hmm = np.random.choice(self.base_hmms)
        return random_hmm  # Copy or initialize new HMM structure as needed.

    def _compute_responsibility(self, base_hmm, cluster_center):
        """
        Compute the responsibility of a cluster center for a base HMM.
        :param base_hmm: A single base HMM.
        :param cluster_center: A cluster center HMM.
        :return: Responsibility score (likelihood-based).
        """
        # Approximate KL divergence via variational lower bound
        kl_div = self._approximate_kl(base_hmm, cluster_center)
        return -kl_div  # Responsibility proportional to negative KL divergence

    def _approximate_kl(self, base_hmm, cluster_center):
        """
        Approximate the KL divergence between two HMMs using variational approximation.
        :param base_hmm: HMM whose distribution is compared.
        :param cluster_center: Representative cluster center HMM.
        :return: Approximate KL divergence.
        """
        # Use synthetic data or variational approximations
        synthetic_data = base_hmm.sample(100)[0]  # Sample data from base HMM
        log_likelihood_base = np.mean(base_hmm.score_samples(synthetic_data))
        log_likelihood_cluster = np.mean(cluster_center.score_samples(synthetic_data))
        return log_likelihood_base - log_likelihood_cluster

    def e_step(self):
        """
        Variational E-step: Compute responsibilities for all base HMMs and clusters.
        """
        for i, base_hmm in enumerate(self.base_hmms):
            responsibilities = []
            for cluster_center in self.cluster_centers:
                responsibility = self._compute_responsibility(base_hmm, cluster_center)
                responsibilities.append(responsibility)
            responsibilities = np.exp(responsibilities - np.max(responsibilities))  # Stabilize softmax
            self.assignments[i] = responsibilities / np.sum(responsibilities)

    def m_step(self):
        """
        Variational M-step: Update cluster centers using the responsibilities.
        """
        for j in range(self.n_clusters):
            # Collect weighted samples for each cluster
            weighted_samples = []
            weights = []
            for i, base_hmm in enumerate(self.base_hmms):
                resp = self.assignments[i, j]
                samples, _ = base_hmm.sample(100)  # Sample data from base HMM
                weighted_samples.append(samples)
                weights.append(resp)
            
            # Combine samples and fit the cluster center HMM
            combined_samples = np.vstack(weighted_samples)
            combined_weights = np.hstack(weights)
            self.cluster_centers[j].fit(combined_samples, combined_weights)

    def fit(self, max_iter=10, tol=1e-3):
        """
        Fit the VHEM algorithm to cluster the HMMs.
        :param max_iter: Maximum number of iterations.
        :param tol: Convergence tolerance.
        """
        prev_assignments = None
        for iteration in range(max_iter):
            self.e_step()
            self.m_step()
            
            # Check convergence
            if prev_assignments is not None:
                assignment_diff = np.linalg.norm(self.assignments - prev_assignments)
                if assignment_diff < tol:
                    print(f"Converged at iteration {iteration}.")
                    break
            prev_assignments = self.assignments.copy()
        else:
            print("Reached maximum iterations without convergence.")