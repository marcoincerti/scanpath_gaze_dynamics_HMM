import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image
import numpy as np
import torch
import matplotlib.cm as cm

# Funzione per plottare i punti di fissazione colorati per ROI previste
def plot_fixations_with_rois(background_image_path, X, predicted_rois, subject_id):
    """
    Plotta i punti di fissazione colorati per le ROI previste sull'immagine di sfondo.

    Args:
        background_image_path (str): Percorso dell'immagine di sfondo.
        X (numpy.ndarray): Punti di fissazione di forma (n_samples, 2).
        predicted_rois (numpy.ndarray): ROI previste di forma (n_samples,).
        subject_id (str or int): Identificatore per il soggetto.
    """
    plt.figure(figsize=(10, 8))

    # Carica l'immagine di sfondo
    if os.path.exists(background_image_path):
        img = plt.imread(background_image_path)
        plt.imshow(img, extent=[0, img.shape[1], img.shape[0], 0])  # Aggiusta l'estensione se necessario
    else:
        print(f"Immagine di sfondo non trovata: {background_image_path}")
        plt.gca().invert_yaxis()  # Inverti l'asse Y se necessario

    # Plot dei punti di fissazione colorati per le ROI previste
    scatter = plt.scatter(X[:, 0], X[:, 1], c=predicted_rois, cmap='viridis', marker='o', edgecolor='k')
    plt.colorbar(scatter, label='ROI Previste')
    plt.title(f"Fissazioni del Soggetto {subject_id} Colorate per ROI Previste")
    plt.xlabel('Coordinata X')
    plt.ylabel('Coordinata Y')
    plt.gca().invert_yaxis()  # Inverti l'asse Y se necessario
    plt.tight_layout()
    plt.show()

def plot_fixations_by_roi(image_path, fixation_data, best_z_preds, model, subject_idx, subject_id, scale_factor=3, save_plot=False):
    """
    Plot all fixation points for the subject, divided into ROIs (regions of interest) based on the model.
    The fixations will be grouped by ROI, and an ellipse will be drawn around each ROI.

    Parameters:
    - image_path: path to the background image
    - fixation_data: list of fixation points for each subject, where each trial has multiple fixation points
    - best_z_preds: predicted ROIs (HMM states) for each fixation point
    - model: trained model that provides the number of ROIs (K)
    - subject_idx: index of the subject whose fixation points are to be plotted
    - subject_id: identifier of the subject (for labeling purposes)
    - scale_factor: scaling factor for the ellipses to make them larger
    - save_plot: if True, the plot will be saved as an image file
    """
    
    # Load the background image (e.g., a face or any visual scene)
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Create the figure with the same size as the image
    fig, ax = plt.subplots(figsize=(img_width / 100, img_height / 100), dpi=100)

    # Show the image as the background
    ax.imshow(img, extent=[0, img_width, img_height, 0])

    # Extract fixation data for the subject
    subject_fixations = fixation_data[subject_idx]['Trials']  # Extract trials list for the subject

    # Concatenate all fixation points for the subject across trials
    X = np.concatenate([trial['Fixations'] for trial in subject_fixations])

    # Predicted ROIs (HMM states) for each fixation point
    predicted_rois = best_z_preds

    # Number of ROIs
    num_rois = model.n_components if hasattr(model, 'n_components') else len(np.unique(predicted_rois))

    # Generate a colormap with enough colors for all ROIs
    roi_colors = cm.get_cmap('tab10', num_rois)

    # Plot fixation points color-coded by ROI
    for roi_idx in range(num_rois):
        # Filter fixation points belonging to the current ROI
        roi_points = X[predicted_rois == roi_idx]
        if len(roi_points) > 0:
            fix_x = roi_points[:, 0]  # X coordinates
            fix_y = roi_points[:, 1]  # Y coordinates

            # Plot fixation points with the corresponding ROI color
            color = roi_colors(roi_idx)
            ax.scatter(fix_x, fix_y, color=color, label=f'ROI {roi_idx+1}', alpha=0.6)

            # Calculate the ellipse that contains the fixation points of this ROI
            mean_x, mean_y = np.mean(fix_x), np.mean(fix_y)
            cov = np.cov(fix_x, fix_y)

            # Eigen decomposition for ellipse parameters
            eigvals, eigvecs = np.linalg.eigh(cov)

            # Check for positive eigenvalues
            if np.any(eigvals <= 0):
                continue  # Skip drawing the ellipse if covariance is not positive definite

            angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigvals)

            # Multiply the width and height by the scale factor
            width *= scale_factor
            height *= scale_factor

            # Draw the ellipse
            ellipse = Ellipse((mean_x, mean_y), width=width, height=height, angle=angle,
                              edgecolor=color, facecolor='none', lw=2)
            ax.add_patch(ellipse)

    # Set axis labels
    ax.set_xlabel('X Axis (FixX)')
    ax.set_ylabel('Y Axis (FixY)')

    # Set axis limits to match the image size
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)  # Invert Y-axis if necessary

    # Add legend outside the plot area
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    # Add title with subject ID
    ax.set_title(f'Subject {subject_id} Fixations Grouped by ROI')

    # Adjust layout
    plt.tight_layout()

    # Save or show the plot
    if save_plot:
        # Save the plot to a file
        plt.savefig(f'subject_{subject_id}_fixations_by_roi.png', bbox_inches='tight')
        plt.close()
    else:
        # Show the plot
        plt.show()

def plot_fixations_with_sequence(image_path, fixation_data, subject_idx, subject_id, save_plot=False):
    """
    Plot all fixation points for the subject, showing the sequence of fixations, with each trial in a different color.
    The first fixation will be marked with an "X", and consecutive fixations will be connected with lines.
    
    Parameters:
    - image_path: path to the background image
    - fixation_data: list of fixation points for each subject, where each trial has multiple fixation points
    - subject_idx: index of the subject whose fixation points are to be plotted
    - subject_id: identifier of the subject (for labeling purposes)
    - save_plot: if True, the plot will be saved as an image file
    """
    
    # Load the image
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Create the figure with the same size as the image
    fig, ax = plt.subplots(figsize=(img_width / 100, img_height / 100), dpi=100)

    # Show the image as the background
    ax.imshow(img, extent=[0, img_width, img_height, 0])

    # Extract fixation data for the subject
    subject_fixations = fixation_data[subject_idx]['Trials']  # Extract trials list for the subject

    # Color map for each trial
    num_trials = len(subject_fixations)
    trial_colors = cm.get_cmap('tab20', num_trials)  # Use a color map with different colors for each trial

    # Loop over trials and plot each trial's fixation points
    for i, trial in enumerate(subject_fixations):
        fixations = trial['Fixations']
        fix_x = fixations[:, 0]  # X coordinates
        fix_y = fixations[:, 1]  # Y coordinates

        # Plot lines connecting the fixation points in sequence for each trial
        ax.plot(fix_x, fix_y, color=trial_colors(i), linestyle='-', marker='', alpha=0.8)

        # Plot fixation points for each trial
        ax.scatter(fix_x[1:], fix_y[1:], color=trial_colors(i), s=10, label=f'Trial {i+1}', zorder=2)  # Smaller dots
        ax.scatter(fix_x[0], fix_y[0], color=trial_colors(i), marker='x', s=40, zorder=3)  # Smaller "X" for the first fixation of each trial

    # Set axis labels
    ax.set_xlabel('X Axis (FixX)')
    ax.set_ylabel('Y Axis (FixY)')

    # Set axis limits to match the image size
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)  # Invert Y-axis if necessary

    # Add title with subject ID
    ax.set_title(f'Subject {subject_id} Fixations and Sequence (per Trial)')

    # Adjust layout
    plt.tight_layout()

    # Save or show the plot
    if save_plot:
        # Save the plot to a file
        plt.savefig(f'subject_{subject_id}_fixations_sequence_per_trial.png', bbox_inches='tight')
        plt.close()
    else:
        # Show the plot
        plt.show()