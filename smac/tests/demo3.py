import numpy as np
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from smac.utils import read_and_convert_txt_to_dataframe, read_fixation_data
from smac.model_training_v2 import train_best_model
from smac.cluster_v2 import vhem_cluster_models
from smac.visualization import visualize_all
from typing import List, Optional

# ==============================
# Main Entry Point
# ==============================

def main():
    # File paths
    #file_path = "/Users/marco.incerti/Desktop/uni-projects/INMCA_Incerti/smac/tests/demodata.xls"
    #image_path = '/Users/marco.incerti/Desktop/uni-projects/INMCA_Incerti/smac/tests/face.jpg'
    #data, subjects = read_fixation_data(file_path)
    
    file_path = "/Users/marco.incerti/Desktop/uni-projects/INMCA_Incerti/smac/tests/Fixations_2023.txt"
    data, subjects = read_and_convert_txt_to_dataframe(file_path)
    image_path = None

    # Parameters
    num_states_list = [2, 3]  # List of state counts for the HMM
    max_iter = 1000  # Maximum number of iterations for model fitting
    models = []

    # Check if data and subjects are valid
    if data.empty:
        logging.error("The data file is empty. Please check the input file.")
        return

    if subjects is None or len(subjects) == 0:
        logging.error("No subjects found in the data. Please verify the data structure.")
        return

    # Step 1: Train individual models for each subject using multi-threading
    models = train_subject_models_multithreaded(data, subjects, num_states_list, max_iter, image_path)

    # Step 2: Cluster models and process groups
    if models:
        cluster_and_process_groups_multithreaded(data, subjects, models, num_states_list, max_iter, image_path)
    else:
        logging.warning("No models were trained successfully.")
        

# ==============================
# Multi-Threaded Model Training
# ==============================
def train_subject_model(subject, data, num_states_list, max_iter):
    """Train an HMM model for a single subject."""
    logging.info(f"Processing Subject {subject}...")
    subject_data = data[data['SubjectID'] == subject]

    sequences = []
    for trial in subject_data['TrialID'].unique():
        trial_data = subject_data[subject_data['TrialID'] == trial]
        sequence = trial_data[['FixX', 'FixY']].values
        sequences.append(sequence)

    if not sequences:
        logging.warning(f"No data available for Subject {subject}.")
        return None

    X = np.concatenate(sequences)
    return train_best_model(X, num_states_list, max_iter=max_iter)

def train_subject_models_multithreaded(data, subjects, num_states_list, max_iter, image_path):
    """Train HMM models for all subjects using multi-threading."""
    models = []
    with ThreadPoolExecutor() as executor:
        future_to_subject = {executor.submit(train_subject_model, subject, data, num_states_list, max_iter): subject for subject in subjects}

        for future in as_completed(future_to_subject):
            subject = future_to_subject[future]
            try:
                best_model = future.result()
                if best_model:
                    models.append(best_model)
                    logging.info(f"Model trained successfully for Subject {subject}.")
                    
                    # Plot and save the visualization for each subject
                    subject_data = data[data['SubjectID'] == subject]
                    visualize_subject_data(subject_data, best_model, image_path, subject)
                else:
                    logging.warning(f"No valid model found for Subject {subject}.")
            except Exception as e:
                logging.error(f"Error training model for Subject {subject}: {e}")
    return models

# ==============================
# Multi-Threaded Group Processing
# ==============================
def process_group(cluster_id, subjects, cluster_labels, data, num_states_list, max_iter, image_path):
    """Process a single group by merging data, training a model, and visualizing results."""
    logging.info(f"Processing Group {cluster_id}...")
    group_data = merge_subjects_in_group(subjects, cluster_labels, cluster_id, data)

    if group_data.empty:
        logging.warning(f"No data available for Group {cluster_id}.")
        return None

    sequences = [group_data[['FixX', 'FixY']].values]
    X = np.concatenate(sequences)

    best_model = train_best_model(X, num_states_list, max_iter=max_iter)
    if best_model:
        logging.info(f"Training completed for Group {cluster_id}.")
        visualize_group_data(group_data, best_model, image_path, cluster_id)
    else:
        logging.warning(f"No valid model found for Group {cluster_id}.")
    return best_model


def cluster_and_process_groups_multithreaded(data, subjects, models, num_states_list, max_iter, image_path):
    """Cluster trained models and process groups using multi-threading."""
    # Add a parameter for the number of clusters
    n_clusters = 2

    # Update the clustering call
    cluster_labels, kmeans = vhem_cluster_models(models, n_clusters=n_clusters)
    logging.info(f"Clustering results: {cluster_labels}")

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_group, cluster_id, subjects, cluster_labels, data, num_states_list, max_iter, image_path)
            for cluster_id in range(n_clusters)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing group: {e}")


# ==============================
# Visualization
# ==============================
def visualize_subject_data(subject_data, model, image_path, subject):
    """Visualize fixation data for a single subject."""
    hidden_states = model.predict(subject_data[['FixX', 'FixY']].values)
    subject_data_with_rois = subject_data.copy()
    subject_data_with_rois['hidden_state'] = hidden_states

    metrics = compute_metrics(subject_data_with_rois)

    # Ensure the plots directory exists
    import os
    os.makedirs("plots/subjects", exist_ok=True)

    # Save the plot for the subject
    visualize_all(
        subject_data_with_rois,
        f"Subject_{subject}",
        image_path,
        metrics['roi_counts'],
        metrics['transition_counts'],
        metrics['transition_matrix'],
        metrics['prior_probs'],
        save_path=f"plots/subjects/Subject_{subject}_visualization.png"
    )
    
def visualize_group_data(group_data, model, image_path, group_id):
    """Visualize fixation data for a group."""
    hidden_states = model.predict(group_data[['FixX', 'FixY']].values)
    group_data_with_rois = group_data.copy()
    group_data_with_rois['hidden_state'] = hidden_states

    metrics = compute_metrics(group_data_with_rois)

    # Ensure the plots directory exists
    import os
    os.makedirs("plots", exist_ok=True)

    visualize_all(
        group_data_with_rois,
        f"Group_{group_id}",
        image_path,
        metrics['roi_counts'],
        metrics['transition_counts'],
        metrics['transition_matrix'],
        metrics['prior_probs'],
        save_path=f"plots/Group_{group_id}_visualization.png"
    )

def compute_metrics(data_with_rois: pd.DataFrame) -> dict:
    """
    Compute ROI metrics including transition counts, prior probabilities, and transition matrix.

    Args:
        data_with_rois (pd.DataFrame): DataFrame containing columns 'TrialID', 'FixX', 'FixY', and 'hidden_state'.

    Returns:
        dict: A dictionary containing the following keys:
            - "roi_counts": Counts of occurrences for each ROI.
            - "transition_counts": Transition counts matrix.
            - "transition_matrix": Transition probability matrix.
            - "prior_probs": Initial state probabilities.
    """
    if data_with_rois.empty:
        return {
            "roi_counts": [],
            "transition_counts": np.array([]),
            "transition_matrix": np.array([]),
            "prior_probs": []
        }

    # Ensure 'hidden_state' column exists
    if 'hidden_state' not in data_with_rois.columns:
        raise ValueError("'hidden_state' column is missing in the input DataFrame.")

    hidden_states = data_with_rois['hidden_state'].values

    # Ensure states are zero-based integers
    if not np.issubdtype(hidden_states.dtype, np.integer) or hidden_states.min() < 0:
        raise ValueError("'hidden_state' column must contain zero-based integers.")

    # ROI counts
    n_states = hidden_states.max() + 1  # Ensure this covers all states
    roi_counts = np.bincount(hidden_states, minlength=n_states)

    # Transition counts
    transition_counts = np.zeros((n_states, n_states), dtype=int)
    for trial_id in data_with_rois['TrialID'].unique():
        trial_states = data_with_rois[data_with_rois['TrialID'] == trial_id]['hidden_state'].values
        for i in range(1, len(trial_states)):
            transition_counts[trial_states[i - 1], trial_states[i]] += 1

    # Transition matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        transition_matrix = np.nan_to_num(
            transition_counts / transition_counts.sum(axis=1, keepdims=True)
        )

    # Prior probabilities (from first states of each trial)
    first_fixations = data_with_rois.groupby('TrialID').first()
    if 'hidden_state' not in first_fixations.columns or first_fixations.empty:
        prior_probs = np.zeros(n_states)
    else:
        first_states = first_fixations['hidden_state'].values
        prior_probs = np.bincount(first_states, minlength=n_states) / len(first_states)

    return {
        "roi_counts": roi_counts.tolist(),
        "transition_counts": transition_counts,
        "transition_matrix": transition_matrix,
        "prior_probs": prior_probs.tolist(),
    }


# ==============================
# Data Merging
# ==============================
def merge_subjects_in_group(subjects, cluster_labels, cluster_id, data):
    """Merge fixation data for all subjects in a specific group."""
    group_subjects = [subject for i, subject in enumerate(subjects) if cluster_labels[i] == cluster_id]
    return data[data['SubjectID'].isin(group_subjects)]


# ==============================
# Script Entry Point
# ==============================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()