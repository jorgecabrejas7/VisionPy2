import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import shift as nd_shift
from pystackreg import StackReg


def mse(shifted, reference):
    """Calculate Mean Squared Error between two images."""
    return np.sum((shifted - reference) ** 2)


def cross_correlation(shifted, reference):
    """Calculate Cross-Correlation between two images."""
    return -np.sum(shifted * reference)


def register_images_intensity(moving, reference, metric="mse"):
    """Use optimization to register two images with a selected metric."""

    metrics = {
        "mse": mse,
        "cross_correlation": cross_correlation,
    }

    if metric not in metrics:
        raise ValueError(
            f"Unknown metric: {metric}. Choose from {list(metrics.keys())}"
        )

    def shift_moving(shift):
        """Apply a shift to the moving image."""
        return nd_shift(moving, shift=shift)

    # Initial guess for the shift
    initial_shift = [0, 0]

    # Define the objective function based on the selected metric
    def objective_function(shift):
        shifted_moving = shift_moving(shift)
        return metrics[metric](shifted_moving, reference)

    # Use 'minimize' from SciPy for optimization
    result = minimize(objective_function, initial_shift, method="Powell")

    # Apply the optimal shift
    optimal_shift = result.x
    registered = shift_moving(optimal_shift)

    return registered, optimal_shift


def stackreg_translate(img1: np.ndarray, img2: np.ndarray):
    """
    Register two images using Python StackReg Plugin implementation.

    Parameters:
        img1 (np.ndarray): The first image to be registered.
        img2 (np.ndarray): The second image to be registered.

    Returns:
        StackReg: The StackReg object containing the registration results.
    """

    sr = StackReg(StackReg.TRANSLATION)
    sr.register(img1, img2)
    return sr
