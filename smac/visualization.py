import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from PIL import Image
import logging


def plot_fixation_sequences(fixation_data, trial_ids, image_path, ax):
    """Top-left plot: Fixation sequences for each trial with different colors, first fixation marked as 'x'."""
    if image_path:
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
            ax.imshow(img, extent=[0, img_width, img_height, 0])
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
    
    # Get unique trial IDs to iterate over
    unique_trials = np.unique(trial_ids)
    colors = plt.cm.get_cmap('tab10', len(unique_trials))  # Use a colormap to get different colors

    for i, trial_id in enumerate(unique_trials):
        trial_data = fixation_data[trial_ids == trial_id]
        ax.plot(
            trial_data[:, 0], trial_data[:, 1],
            marker='o', linestyle='-', color=colors(i),
            label=f'Trial {trial_id}'
        )
        # Mark the first fixation of each trial
        ax.scatter(
            trial_data[0, 0], trial_data[0, 1],
            color=colors(i), marker='x', s=100, label=f'First Fixation Trial {trial_id}'
        )

    ax.set_title("Fixation Sequences")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def plot_hidden_states(subject_data, subject, image_path, ax):
    """Top-middle plot: Hidden states with ellipses enclosing the ROIs over a background image if provided."""
    if image_path:
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
            ax.imshow(img, extent=[0, img_width, img_height, 0])
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
    
    scatter = ax.scatter(
        subject_data['FixX'],
        subject_data['FixY'],
        c=subject_data['hidden_state'],
        cmap='viridis',
        s=30,
        alpha=0.7
    )

    hidden_states = subject_data['hidden_state'].unique()
    colormap = plt.cm.get_cmap('viridis')

    for state in hidden_states:
        state_data = subject_data[subject_data['hidden_state'] == state]
        if len(state_data) > 1:
            mean = state_data[['FixX', 'FixY']].mean().values
            cov = np.cov(state_data[['FixX', 'FixY']].values.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
            width, height = 4 * np.sqrt(eigvals)
            color = colormap(state / len(hidden_states))

            ellipse = Ellipse(
                xy=mean,
                width=width,
                height=height,
                angle=angle,
                edgecolor=color,
                facecolor='none',
                linewidth=2,
                alpha=1.0
            )
            ax.add_patch(ellipse)

    ax.set_xlabel('FixX')
    ax.set_ylabel('FixY')
    ax.set_title(f'Subject {subject} - Predicted Hidden States')
    
    plt.colorbar(scatter, ax=ax, label='Hidden State')


def plot_roi_counts(roi_counts, ax):
    """Top-right plot: Total number of fixations in each ROI."""
    ax.bar(range(len(roi_counts)), roi_counts, color='skyblue')
    ax.set_title("Number of Fixations in Each ROI")
    ax.set_xlabel("ROI")
    ax.set_ylabel("Count")


def plot_transition_counts(transition_counts, ax):
    """Bottom-left plot: Transition counts between ROIs."""
    ax.imshow(transition_counts, cmap='Blues', aspect='auto')
    ax.set_title("Transition Counts")
    ax.set_xlabel("To ROI")
    ax.set_ylabel("From ROI")


def plot_transition_matrix(transition_matrix, ax):
    """Bottom-middle plot: Normalized transition matrix."""
    normalized_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    ax.imshow(normalized_matrix, cmap='Reds', aspect='auto')
    ax.set_title("Transition Matrix (Normalized)")
    ax.set_xlabel("To ROI")
    ax.set_ylabel("From ROI")


def plot_prior_probabilities(prior_probs, ax):
    """Bottom-right plot: Prior probability of the ROI of the first fixation."""
    ax.bar(range(len(prior_probs)), prior_probs, color='lightcoral')
    ax.set_title("Prior Probability (First Fixation)")
    ax.set_xlabel("ROI")
    ax.set_ylabel("Probability")


def visualize_all(data, subject, image_path, roi_counts, transition_counts, transition_matrix, prior_probs, save_path=None):
    """Visualizes all plots in a single view, extracting information from `data`."""
    
    # Extract fixation data and trial IDs from the data
    fixation_data = data[['FixX', 'FixY']].values
    trial_ids = data['TrialID'].values  # Assuming `TrialID` is a column in your data

    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    # Top-left: Fixation sequences
    plot_fixation_sequences(fixation_data, trial_ids, image_path, axs[0, 0])
    
    # Top-middle: Hidden states on the background image
    plot_hidden_states(data, subject, image_path, axs[0, 1])
    
    # Top-right: Total number of fixations in each ROI
    plot_roi_counts(roi_counts, axs[0, 2])
    
    # Bottom-left: Transition counts
    plot_transition_counts(transition_counts, axs[1, 0])
    
    # Bottom-middle: Transition matrix
    plot_transition_matrix(transition_matrix, axs[1, 1])
    
    # Bottom-right: Prior probability
    plot_prior_probabilities(prior_probs, axs[1, 2])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logging.info(f"Saved plot to {save_path}")
    else:
        logging.warning("No save path provided; skipping save operation.")

    plt.close(fig)