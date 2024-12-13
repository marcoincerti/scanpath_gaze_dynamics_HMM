import logging
import os
import numpy as np
from smac.utils import read_fixation_data
from smac.model_training import train_hmm
from smac.prediction import predict_rois
from smac.visualization import visualize_all
from smac.cluster import vhem_cluster_models, VHEMClustering

def main():
    """Main function to execute the script with predefined paths."""
    # Set file paths
    file_path = '/Users/marco.incerti/Desktop/uni-projects/INMCA_Incerti/smac/tests/demodata.xls'
    image_path = '/Users/marco.incerti/Desktop/uni-projects/INMCA_Incerti/smac/tests/face.jpg'
    num_states_list = [2, 3]  # List of state counts to try for the HMM
    n_inits = 20  # Number of initializations for model training
    max_iter = 1000  # Maximum number of iterations for model fitting

    # Check if files exist
    if not os.path.isfile(file_path):
        logging.error(f"Data file {file_path} does not exist.")
        return
    if not os.path.isfile(image_path):
        logging.error(f"Image file {image_path} does not exist.")
        return

    # Read data
    data, subjects = read_fixation_data(file_path)
    models = []

    if data is not None and subjects is not None:
        for subject in subjects:
            logging.info(f"Processing Subject {subject}...")

            # Train models and select the best one
            best_model_tuple = train_hmm(
                subject, data, num_states_list, n_inits=n_inits, max_iter=max_iter
            )

            if best_model_tuple:
                models.append(best_model_tuple[0])  # Collect trained models

                # Predict ROIs using the best model
                subject_data_with_rois = predict_rois(subject, data, best_model_tuple)
                if subject_data_with_rois is not None:
                    # Prepare the data for visualization
                    fixation_data = subject_data_with_rois[['FixX', 'FixY']].values
                    hidden_states = subject_data_with_rois['hidden_state'].values

                    # Calculate ROI counts
                    roi_counts = np.bincount(hidden_states)

                    # Calculate transition counts
                    n_states = len(np.unique(hidden_states))
                    transition_counts = np.zeros((n_states, n_states))
                    for trial_id in subject_data_with_rois['TrialID'].unique():
                        trial_data = subject_data_with_rois[subject_data_with_rois['TrialID'] == trial_id]
                        trial_states = trial_data['hidden_state'].values
                        for i in range(1, len(trial_states)):
                            transition_counts[trial_states[i - 1], trial_states[i]] += 1

                    # Normalize to get the transition matrix (handle division by zero)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        transition_matrix = np.nan_to_num(
                            transition_counts / transition_counts.sum(axis=1, keepdims=True)
                        )

                    # Calculate prior probabilities for the first fixation in each ROI
                    first_fixations = subject_data_with_rois.groupby('TrialID').first()
                    first_states = first_fixations['hidden_state'].values
                    prior_probs = np.bincount(first_states, minlength=n_states)
                    prior_probs = prior_probs / prior_probs.sum()

                    # Visualize all plots
                    '''
                    visualize_all(
                        subject_data_with_rois, subject, image_path,
                        roi_counts, transition_counts, transition_matrix, prior_probs
                    )
                    '''
            else:
                logging.warning(f"No valid model found for Subject {subject}.")
                
        # Perform clustering on the collected models using VHEM
        cluster_labels, kmeans = vhem_cluster_models(models, n_clusters=2)      
        logging.info(f"Clustering results: {cluster_labels}")
        
        '''
        n_clusters = 2
        vhem = VHEMClustering(models, n_clusters)

        # Run the clustering
        vhem.fit(max_iter=20, tol=1e-4)

        # Print cluster assignments
        print("Cluster Assignments:")
        for i, responsibilities in enumerate(vhem.assignments):
            assigned_cluster = np.argmax(responsibilities)
            print(f"Base Model {i}: Assigned to Cluster {assigned_cluster} (Responsibility: {responsibilities[assigned_cluster]:.2f})")
        '''

    else:
        logging.error("Data could not be read. Exiting.")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()