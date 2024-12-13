import numpy as np
from scipy.special import psi, gammaln
from sklearn.preprocessing import StandardScaler
import logging


class VariationalBayesianHMM:
    def __init__(self, num_states, dim_obs, observations, alpha_prior=1.0, wishart_params=None):
        self.K = num_states  # Number of hidden states
        self.D = dim_obs  # Dimensionality of observations
        self.alpha_prior = alpha_prior  # Dirichlet prior for transition probabilities

        # Initialize variational parameters for transition probabilities
        self.alpha = np.full((self.K, self.K), self.alpha_prior, dtype=np.float64)

        # Dirichlet prior for initial state probabilities
        self.pi_prior = np.ones(self.K, dtype=np.float64)
        self.pi = np.copy(self.pi_prior)

        # Prior parameters for Gaussian-Wishart distribution
        if wishart_params is None:
            self.beta0 = 1.0
            self.m0 = np.zeros(self.D)
            self.W0 = np.eye(self.D)
            self.nu0 = self.D + 2
        else:
            self.beta0 = wishart_params['beta0']
            self.m0 = wishart_params['m0']
            self.W0 = wishart_params['W0']
            self.nu0 = wishart_params['nu0']

        # Initialize variational parameters for Gaussian-Wishart distribution
        self.beta = np.full(self.K, self.beta0, dtype=np.float64)
        self.m = np.random.normal(size=(self.K, self.D))  # Random initialization for means
        self.W = np.tile(self.W0, (self.K, 1, 1))  # Start with prior W
        self.nu = np.full(self.K, self.nu0, dtype=np.float64)

        # Scale observations
        self.scaler = StandardScaler()
        self.observations_scaled = self.scaler.fit_transform(observations)

    def _e_step(self):
        T = len(self.observations_scaled)
        ln_pi_hat = psi(self.pi) - psi(np.sum(self.pi))  # Initial state probabilities
        ln_alpha_hat = psi(self.alpha) - psi(np.sum(self.alpha, axis=1, keepdims=True))  # Transition probabilities

        # Compute expected log emission probabilities
        E_ln_lambda = np.zeros(self.K)
        E_ln_prob = np.zeros((T, self.K))
        for k in range(self.K):
            nu_term = (self.nu[k] + 1 - np.arange(1, self.D + 1)) / 2
            E_ln_lambda[k] = np.sum(psi(np.maximum(nu_term, 1e-10))) + \
                             self.D * np.log(2) + np.linalg.slogdet(self.W[k])[1]
            diff = self.observations_scaled - self.m[k]
            term = np.einsum('ij,ij->i', diff @ self.W[k], diff)
            E_ln_prob[:, k] = 0.5 * (E_ln_lambda[k] - self.D / self.beta[k] - self.nu[k] * term - self.D * np.log(2 * np.pi))

        # Forward and backward passes
        ln_forward = np.zeros((T, self.K))
        ln_backward = np.zeros((T, self.K))

        ln_forward[0] = ln_pi_hat + E_ln_prob[0]
        for t in range(1, T):
            for j in range(self.K):
                ln_forward[t, j] = E_ln_prob[t, j] + np.logaddexp.reduce(
                    ln_forward[t - 1] + ln_alpha_hat[:, j]
                )

        ln_backward[-1] = 0
        for t in reversed(range(T - 1)):
            for i in range(self.K):
                ln_backward[t, i] = np.logaddexp.reduce(
                    ln_alpha_hat[i] + E_ln_prob[t + 1] + ln_backward[t + 1]
                )

        # Compute responsibilities
        ln_gamma = ln_forward + ln_backward
        max_ln_gamma = np.max(ln_gamma, axis=1, keepdims=True)
        ln_gamma = ln_gamma - max_ln_gamma - np.log(
            np.sum(np.exp(ln_gamma - max_ln_gamma), axis=1, keepdims=True)
        )
        gamma = np.exp(ln_gamma)

        xi = np.zeros((T - 1, self.K, self.K))
        for t in range(T - 1):
            for i in range(self.K):
                xi[t, i] = ln_forward[t, i] + ln_alpha_hat[i] + \
                           E_ln_prob[t + 1] + ln_backward[t + 1]
            xi[t] -= np.logaddexp.reduce(xi[t].flatten())
            xi[t] = np.exp(xi[t])

        return gamma, xi

    def _m_step(self, gamma, xi):
        T = len(self.observations_scaled)
        Nk = np.sum(gamma, axis=0)  # Sufficient statistics for each state
        Nk_safe = np.maximum(Nk, 1e-5)  # Avoid divide-by-zero issues

        # Update initial state probabilities
        self.pi = self.pi_prior + gamma[0]

        # Update transition probabilities
        sum_xi = np.sum(xi, axis=0)
        self.alpha = self.alpha_prior + sum_xi

        # Update Gaussian-Wishart parameters
        for k in range(self.K):
            self.beta[k] = self.beta0 + Nk_safe[k]
            x_bar = np.sum(self.observations_scaled * gamma[:, k][:, np.newaxis], axis=0) / Nk_safe[k]
            self.m[k] = (self.beta0 * self.m0 + Nk_safe[k] * x_bar) / self.beta[k]

            diff = self.observations_scaled - x_bar
            S = np.einsum('i,ij,ik->jk', gamma[:, k], diff, diff)
            m_diff = x_bar - self.m0
            W_inv_matrix = np.linalg.inv(self.W0) + S + \
                           (self.beta0 * Nk_safe[k] / self.beta[k]) * np.outer(m_diff, m_diff)
            self.W[k] = np.linalg.inv(W_inv_matrix + 1e-4 * np.eye(self.D))
            self.nu[k] = max(self.nu0 + Nk_safe[k], self.D + 1)

    def fit(self, max_iter=100, tol=1e-4):
        prev_elbo = -np.inf
        for iteration in range(max_iter):
            gamma, xi = self._e_step()
            self._m_step(gamma, xi)

            # Compute ELBO
            elbo = self._compute_elbo(gamma, xi)
            if np.abs(elbo - prev_elbo) < tol:
                logging.info(f"Converged at iteration {iteration}.")
                break
            prev_elbo = elbo

            logging.info(f"Iteration {iteration} ELBO: {elbo:.4f}")
    
    def _compute_elbo(self, gamma, xi):
        elbo = 0
        T = len(self.observations_scaled)

        # Contribution from transition probabilities
        E_ln_pi = psi(self.alpha) - psi(np.sum(self.alpha, axis=1, keepdims=True))
        elbo += np.sum(xi * E_ln_pi)

        # Contribution from emission probabilities
        for k in range(self.K):
            nu_term = (self.nu[k] + 1 - np.arange(1, self.D + 1)) / 2
            E_ln_lambda = np.sum(psi(np.maximum(nu_term, 1e-10))) + \
                          self.D * np.log(2) + np.linalg.slogdet(self.W[k])[1]
            diff = self.observations_scaled - self.m[k]
            term = np.einsum('ij,ij->i', diff @ self.W[k], diff)
            elbo += np.sum(gamma[:, k] * (E_ln_lambda - self.D / self.beta[k] - self.nu[k] * term - self.D * np.log(2 * np.pi)))

        return elbo

    def predict(self):
        gamma, _ = self._e_step()
        return np.argmax(gamma, axis=1)

    def get_parameters(self):
        trans_probs = self.alpha / np.sum(self.alpha, axis=1, keepdims=True)
        gaussians = [{
            'mean': self.m[k],
            'cov': np.linalg.inv(self.nu[k] * self.W[k])
        } for k in range(self.K)]
        return trans_probs, gaussians

    def calculate_log_likelihood(self, observations):
        # Compute the log-likelihood of the observations under the model
        observations_scaled = self.scaler.transform(observations)
        T = len(observations_scaled)
        log_likelihood = 0.0

        # Compute expected log emission probabilities
        E_ln_lambda = np.zeros(self.K)
        E_ln_prob = np.zeros((T, self.K))
        for k in range(self.K):
            nu_term = (self.nu[k] + 1 - np.arange(1, self.D + 1)) / 2
            E_ln_lambda[k] = np.sum(psi(np.maximum(nu_term, 1e-10))) + \
                             self.D * np.log(2) + np.linalg.slogdet(self.W[k])[1]
            diff = observations_scaled - self.m[k]
            term = np.einsum('ij,ij->i', diff @ self.W[k], diff)
            E_ln_prob[:, k] = 0.5 * (E_ln_lambda[k] - self.D / self.beta[k] - self.nu[k] * term - self.D * np.log(2 * np.pi))

        # Forward algorithm to compute log-likelihood
        ln_alpha_hat = psi(self.alpha) - psi(np.sum(self.alpha, axis=1, keepdims=True))

        ln_forward = np.zeros((T, self.K))
        ln_forward[0] = E_ln_prob[0]
        for t in range(1, T):
            for j in range(self.K):
                ln_forward[t, j] = E_ln_prob[t, j] + np.logaddexp.reduce(
                    ln_forward[t - 1] + ln_alpha_hat[:, j]
                )

        log_likelihood = np.logaddexp.reduce(ln_forward[-1])
        return log_likelihood

def train_best_model(data, num_states_list, max_iter=1000):
    """Trains models with different numbers of states and selects the best one."""
    best_model = None
    best_score = -np.inf  # Initialize to negative infinity for maximization
    observations = np.array(data)
    dim_obs = observations.shape[1]

    for num_states in num_states_list:
        logging.info(f"Training VariationalGaussianHMM with {num_states} states.")
        vhmm = VariationalBayesianHMM(
            num_states=num_states,
            dim_obs=dim_obs,
            observations=observations
        )
        vhmm.fit(max_iter=max_iter)

        # Compute a score to select the best model (e.g., log-likelihood)
        try:
            score = vhmm.calculate_log_likelihood(observations)
            logging.info(f"Model with {num_states} states has log-likelihood: {score}")
        except Exception as e:
            logging.warning(
                f"Failed to compute log-likelihood for model with {num_states} states: {e}"
            )
            continue

        if score > best_score:
            best_score = score
            best_model = vhmm

    if best_model is not None:
        logging.info(f"Best model selected with {best_model.K} states.")
        return best_model
    else:
        logging.warning("No valid model was found.")
        return None