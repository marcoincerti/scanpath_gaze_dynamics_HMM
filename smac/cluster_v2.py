from sklearn.cluster import KMeans
import numpy as np
import logging

def extract_model_features(model, max_components=10):
    """
    Extracts and flattens the parameters of a VariationalGaussianHMM model,
    padding to a fixed number of components.

    Parameters:
        model: Trained VariationalGaussianHMM model.
        max_components (int): The maximum number of components to pad to.

    Returns:
        features (np.ndarray): Flattened and padded feature vector.
    """
    # Extract transition probabilities and emission parameters
    trans_probs, gaussians = model.get_parameters()

    # Extract the number of components and feature dimension
    num_components = len(gaussians)
    feature_dim = model.D  # Dimensionality of the observations

    # Limit the number of components to max_components
    num_components = min(num_components, max_components)

    # Sort the components by their stationary probabilities
    # Estimate stationary distribution
    eigvals, eigvecs = np.linalg.eig(trans_probs.T)
    stat_dist = np.real(eigvecs[:, np.isclose(eigvals, 1)])
    stat_dist = stat_dist / stat_dist.sum()
    stat_dist = stat_dist.flatten()

    sorted_indices = np.argsort(-stat_dist)[:num_components]

    # Extract and sort weights, means, and covariances
    weights = stat_dist[sorted_indices]
    means = np.array([gaussians[i]['mean'] for i in sorted_indices])
    covariances = np.array([gaussians[i]['cov'] for i in sorted_indices])

    # Flatten means and covariances
    means_flat = means.flatten()
    covariances_flat = covariances.reshape(num_components, -1).flatten()

    # Pad weights, means, and covariances to ensure a fixed feature size
    weights_padded = np.pad(weights, (0, max_components - num_components), 'constant')
    means_padded = np.pad(means_flat, (0, max_components * feature_dim - means_flat.size), 'constant')
    cov_size = covariances_flat.size
    cov_expected_size = max_components * feature_dim * feature_dim
    covariances_padded = np.pad(covariances_flat, (0, cov_expected_size - cov_size), 'constant')

    # Concatenate all features into a single vector
    features = np.concatenate([weights_padded, means_padded, covariances_padded])
    return features

def vhem_cluster_models(models, n_clusters=2, max_components=10):
    """
    Clusters the VariationalGaussianHMM models into groups using a KMeans-like approach.

    Parameters:
        models (list): List of trained VariationalGaussianHMM models.
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
        except Exception as e:
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
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(feature_matrix_scaled)
    
    return cluster_labels, kmeans

def sample(self, n_samples):
    """
    Generates synthetic data by sampling from the HMM.

    Parameters:
        n_samples (int): Number of samples to generate.

    Returns:
        observations (np.ndarray): Generated observations.
        states (np.ndarray): Hidden states corresponding to the observations.
    """
    trans_probs = self.alpha / np.sum(self.alpha, axis=1, keepdims=True)
    gaussians = [{'mean': self.m[k], 'cov': np.linalg.inv(self.nu[k] * self.W[k])} for k in range(self.K)]

    # Initialize arrays
    observations = np.zeros((n_samples, self.D))
    states = np.zeros(n_samples, dtype=int)

    # Start state (randomly or based on stationary distribution)
    current_state = np.random.choice(self.K)
    states[0] = current_state
    observations[0] = np.random.multivariate_normal(gaussians[current_state]['mean'], gaussians[current_state]['cov'])

    for t in range(1, n_samples):
        current_state = np.random.choice(self.K, p=trans_probs[current_state])
        states[t] = current_state
        observations[t] = np.random.multivariate_normal(gaussians[current_state]['mean'], gaussians[current_state]['cov'])

    return observations, states

def score_samples(self, observations):
    """
    Computes the log-likelihood of the observations under the model.

    Parameters:
        observations (np.ndarray): Observations to compute the log-likelihood for.

    Returns:
        log_likelihoods (np.ndarray): Log-likelihoods of the observations.
    """
    observations_scaled = self.scaler.transform(observations)
    T = len(observations_scaled)

    # Compute expected log emission probabilities
    E_ln_lambda = np.zeros(self.K)
    log_prob = np.zeros((T, self.K))
    for k in range(self.K):
        nu_term = (self.nu[k] + 1 - np.arange(1, self.D + 1)) / 2
        E_ln_lambda[k] = np.sum(psi(np.maximum(nu_term, 1e-10))) + \
                         self.D * np.log(2) + np.linalg.slogdet(self.W[k])[1]
        diff = observations_scaled - self.m[k]
        term = np.einsum('ij,ij->i', diff @ self.W[k], diff)
        log_prob[:, k] = 0.5 * (E_ln_lambda[k] - self.D / self.beta[k] - self.nu[k] * term - self.D * np.log(2 * np.pi))

    # Compute log-likelihoods
    ln_alpha_hat = psi(self.alpha) - psi(np.sum(self.alpha, axis=1, keepdims=True))
    ln_forward = np.zeros((T, self.K))
    ln_forward[0] = log_prob[0]
    for t in range(1, T):
        for j in range(self.K):
            ln_forward[t, j] = log_prob[t, j] + np.logaddexp.reduce(
                ln_forward[t - 1] + ln_alpha_hat[:, j]
            )

    log_likelihood = np.logaddexp.reduce(ln_forward[-1])
    return log_likelihood * np.ones(T)  # Return the same log-likelihood for each sample

class VHEMClustering:
    def __init__(self, base_hmms, n_clusters):
        """
        Variational HEM clustering for HMMs.
        :param base_hmms: List of pre-trained VariationalGaussianHMM models.
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
        :param cluster_center: Representative cluster center HMM.
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
        synthetic_data, _ = base_hmm.sample(100)  # Sample data from base HMM
        log_likelihood_base = base_hmm.calculate_log_likelihood(synthetic_data)
        log_likelihood_cluster = cluster_center.calculate_log_likelihood(synthetic_data)
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
            # Collect weighted observations for each cluster
            weighted_observations = []
            weights = []
            for i, base_hmm in enumerate(self.base_hmms):
                resp = self.assignments[i, j]
                samples, _ = base_hmm.sample(100)  # Sample data from base HMM
                weighted_observations.append(samples)
                weights.append(np.full(samples.shape[0], resp))
            
            # Combine samples and weights
            combined_samples = np.vstack(weighted_observations)
            combined_weights = np.hstack(weights)
            # Refit the cluster center HMM
            self.cluster_centers[j].fit(combined_samples, max_iter=10)

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
                    logging.info(f"Converged at iteration {iteration}.")
                    break
            prev_assignments = self.assignments.copy()
        else:
            logging.info("Reached maximum iterations without convergence.")

    def get_cluster_labels(self):
        """
        Assigns each base HMM to the cluster with the highest responsibility.

        Returns:
            cluster_labels (np.ndarray): Cluster labels for each base HMM.
        """
        return np.argmax(self.assignments, axis=1)