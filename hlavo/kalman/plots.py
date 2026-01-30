import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import attrs

@attrs.define
class RichardsSolverOutput:
    """
    Output class for Richards Equation Solver.

    Attributes:
    ----------
    times : np.ndarray
        Array of output times.
    H : np.ndarray, times in rows.
        Pressure head array at output times.
    moisture : np.ndarray
        Moisture content (absolute saturation) at output times.
    top_flux : np.ndarray
        Influx at the top boundary at output times.
    bottom_flux : np.ndarray
        Influx at the bottom boundary at output times.
    """
    times: np.ndarray
    H: np.ndarray
    moisture: np.ndarray
    top_flux: np.ndarray
    bottom_flux: np.ndarray
    nodes_z: np.ndarray


def plot_richards_output(output, obs_points=None, fname=None, show=False):
    """
    Plot the results from RichardsSolverOutput.

    Parameters:
    -----------
    output : RichardsSolverOutput
        The solver output containing times, pressure head (H), moisture, and fluxes.
    z_coords : list
        List of Z coordinates for observation points to include in the line plots.
    """
    times = output.times
    z_coords = output.nodes_z
    H = output.H
    moisture = output.moisture
    top_flux = output.top_flux
    bottom_flux = output.bottom_flux
    if obs_points is None:
        step = int(len(z_coords) / 10)    # 10 obs. points
        obs_points = z_coords[0:None:step]

    # Prepare the Z coordinate indices for observation
    i_obs_points = [np.argmin(np.abs(z_obs - z_coords)) for z_obs in obs_points]

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))


    # Top-left: Color plot of H vs time and depth
    c1 = axs[0, 0].imshow(H.T, aspect='auto', extent=[times[0], times[-1], z_coords[0], z_coords[-1]],
                          origin='lower', cmap=plt.cm.viridis_r)
    fig.colorbar(c1, ax=axs[0, 0])
    axs[0, 0].set_title('Pressure Head (H)')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Depth')

    # Top-right: Color plot of Moisture vs time and depth
    # Use the reversed viridis colormap
    sat_cmap = plt.cm.viridis_r  # reversed viridis
    sat_norm = matplotlib.colors.Normalize(vmin=0, vmax=0.5)  # linear normalization

    c2 = axs[0, 1].imshow(moisture.T, aspect='auto', extent=[times[0], times[-1], z_coords[0], z_coords[-1]],
                           origin='lower', cmap=sat_cmap, norm=sat_norm)
    fig.colorbar(c2, ax=axs[0, 1])
    axs[0, 1].set_title('Moisture Content')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Depth')

    # Middle-left: Line plot of H at selected observation points
    for idx in i_obs_points:
        axs[1, 0].plot(times, H[:, idx], label=f'z={z_coords[idx]:.2f}')
    axs[1, 0].legend()
    axs[1, 0].set_title('Pressure Head (H) at Observation Points')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Pressure Head (H)')

    # Middle-right: Line plot of Moisture at selected observation points
    for idx in i_obs_points:
        axs[1, 1].plot(times, moisture[:, idx], label=f'Depth={z_coords[idx]:.2f}')
    axs[1, 1].legend()
    axs[1, 1].set_title('Moisture at Observation Points')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Moisture Content')

    if not (top_flux is None or bottom_flux is None):
        # Bottom row: Line plot of top and bottom fluxes
        axs[2, 0].plot(times, top_flux, label='Top Flux')
        axs[2, 0].plot(times, bottom_flux, label='Bottom Flux')
        axs[2, 0].legend()
        axs[2, 0].set_title('Boundary Fluxes')
        axs[2, 0].set_xlabel('Time')
        axs[2, 0].set_ylabel('Flux')

    axs[2, 1].axis('off')  # Leave the last subplot blank

    fig.tight_layout()
    if fname is None:
        fname = 'richards_solver_output.pdf'
    fig.savefig(fname)
    if show:
        plt.show()

def covariance_plot(cov_matrix, time, state_struct, n_evec=10, fname=None, show=False):
    """
    Plots analysis of a given covariance matrix:
    1. Variances (log scale) with grouped labels.
    2. Correlation matrix heatmap (signed log scale colormap) with grouped labels.
    3. Sorted eigenvalues.
    4. First K eigenvectors as a function of index.

    Parameters:
        cov_matrix (numpy.ndarray): The covariance matrix.
        time (float or int): Time index for the title.
        state_struct (dict): Dictionary where keys are group names and values are objects with a `size()` method.
        n_evec (int): Number of eigenvectors to plot.
        fname (str, optional): Filename to save the figure. Defaults to "covariance_analysis_<time>.pdf".
        show (bool, optional): Whether to display the plot.
    """
    # Extract group sizes and boundaries
    group_labels = list(state_struct.keys())
    group_sizes = [state_struct[key].size() for key in group_labels]
    boundaries = np.cumsum([0] + group_sizes)

    # Validate matrix size
    total_size = sum(group_sizes)
    if cov_matrix.shape[0] != total_size or cov_matrix.shape[1] != total_size:
        raise ValueError("Covariance matrix size does not match total size defined in state_struct.")

    title = f'Covariance Analysis at time {time}'
    if fname is None:
        fname = f'covariance_analysis_{time}.pdf'

    # Compute variances (diagonal elements)
    variances = np.diag(cov_matrix)

    # Compute correlation matrix
    std_dev = np.sqrt(variances)
    correlation_matrix = cov_matrix / np.outer(std_dev, std_dev)

    # Compute Eigenvalues and Eigenvectors
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)  # Ensures sorted order (ascending)
    eigvals_sorted = eigvals[::-1]  # Descending order
    eigvecs_sorted = eigvecs[:, ::-1]  # Reorder eigenvectors accordingly

    # Set up figure layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Variance bar plot (log scale)
    ax_var = axes[0, 0]
    ax_var.bar(range(len(variances)), variances, color='blue', alpha=0.7)
    ax_var.set_yscale('log')
    ax_var.set_title('Variances (Log Scale)')
    ax_var.set_xlabel('State Groups')
    ax_var.set_ylabel('Variance (log scale)')

    # Adjust group separators (shifted by 0.5 to fall between bars)
    for boundary in boundaries[1:-1]:  # Avoid first (0) and last (total size)
        ax_var.axvline(boundary - 0.5, color='black', linestyle='--', linewidth=1)  # Shift left by 0.5

    # Set ticks at the center of each group
    ax_var.set_xticks([(boundaries[i] + boundaries[i+1] - 1) / 2 for i in range(len(group_labels))])
    ax_var.set_xticklabels(group_labels, rotation=45, ha='right')

    # 2. Correlation matrix heatmap (signed log scale colormap)
    ax_corr = axes[0, 1]
    signed_log_corr = np.sign(correlation_matrix) * np.log1p(np.abs(correlation_matrix))
    sns.heatmap(signed_log_corr, ax=ax_corr, center=0, cmap='coolwarm', annot=False, xticklabels=False, yticklabels=False)
    ax_corr.set_title('Correlation Matrix (Signed Log Scale)')

    # Add group separators on heatmap
    for boundary in boundaries[1:-1]:
        ax_corr.axvline(boundary, color='white', linestyle='--', linewidth=1.5)
        ax_corr.axhline(boundary, color='white', linestyle='--', linewidth=1.5)

    # Set custom ticks for state groups
    ax_corr.set_xticks([(boundaries[i] + boundaries[i+1]) / 2 for i in range(len(group_labels))])
    ax_corr.set_xticklabels(group_labels, rotation=45, ha='right')
    ax_corr.set_yticks([(boundaries[i] + boundaries[i+1]) / 2 for i in range(len(group_labels))])
    ax_corr.set_yticklabels(group_labels, rotation=0, va='center')

    # 3. Sorted Eigenvalues plot
    ax_eigvals = axes[1, 0]
    ax_eigvals.plot(eigvals_sorted, 'o-', color='red', markersize=5)
    ax_eigvals.set_title('Sorted Eigenvalues')
    ax_eigvals.set_xlabel('Index')
    ax_eigvals.set_ylabel('Eigenvalue')

    # 4. First K Eigenvectors as functions of index i
    ax_eigvecs = axes[1, 1]
    for i in range(min(n_evec, eigvecs_sorted.shape[1])):
        ax_eigvecs.plot(eigvecs_sorted[:, i], label=f'Eigenvector {i + 1}')

    ax_eigvecs.set_title(f'First {n_evec} Eigenvectors')
    ax_eigvecs.set_xlabel('Index')
    ax_eigvecs.set_ylabel('Component Value')
    ax_eigvecs.legend()

    fig.tight_layout()
    fig.suptitle(title)
    fig.savefig(fname)
    if show:
        plt.show()
    plt.close('all')  # Close the figure to prevent overlapping plots