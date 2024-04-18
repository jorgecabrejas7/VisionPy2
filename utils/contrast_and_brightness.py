import numpy as np
import cv2
from utils.bit_depth import imagej_8B

def auto_adjust(image: np.ndarray) -> np.ndarray:
    """
    Automatically adjusts the contrast of an image by stretching its histogram.

    This function assumes the input is a grayscale image in the form of a NumPy array.
    It calculates the image's histogram, determines the lower and upper intensity bounds
    where a significant number of pixels lie, and stretches the histogram across the
    available range (0-255) to enhance contrast.

    Parameters:
    image (Union[np.ndarray, Any]): The input grayscale image as a NumPy array.

    Returns:
    np.ndarray: The contrast-adjusted image as a NumPy array.
    """
    # Validate input is a NumPy array
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    # Calculate histogram and total pixel count
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    pixel_count = image.size

    # Determine the auto threshold
    auto_threshold = 5000
    threshold = pixel_count // auto_threshold

    # Find lower bound
    hmin = next(i for i, count in enumerate(hist) if count > threshold)

    # Find upper bound
    hmax = next(255 - i for i, count in enumerate(reversed(hist)) if count > threshold)

    # Apply contrast stretching and return the result
    return np.clip((image - hmin) * (255 / (hmax - hmin)), 0, 255)


def adjust_brightness_contrast_old(
    image: np.ndarray, min_val: int, max_val: int
) -> np.ndarray:
    """
    Adjusts the contrast of an image based on specified minimum and maximum values.

    Parameters:
    image (np.ndarray): The input grayscale image as a NumPy array.
    min_val (int): The minimum pixel intensity value for contrast stretching.
    max_val (int): The maximum pixel intensity value for contrast stretching.

    Returns:
    np.ndarray: The contrast-adjusted image.
    """
    # Validate input is a NumPy array
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    # Clip and rescale the intensity values of the image
    
    adjusted_image = (image - min_val) / (max_val - min_val) * 255
    clipped_image = np.clip(adjusted_image, 0, 255)

    return clipped_image.astype(np.uint8)

def adjust_brightness_contrast(image: np.ndarray, min_val: int, max_val: int) -> np.ndarray:
    """
    Rescales an image's pixel values to a new specified range [min_val, max_val].

    Parameters:
        image (np.ndarray): The input image array.
        min_val (int or float): The new minimum value for the image pixels.
        max_val (int or float): The new maximum value for the image pixels.

    Returns:
        np.ndarray: The rescaled image array.
    """
    # Determine the current minimum and maximum values in the image
    current_min = np.min(image)
    current_max = np.max(image)

    # Calculate scale and shift to adjust the pixel range
    scale = (max_val - min_val) / (current_max - current_min)
    shift = min_val - current_min * scale

    # Apply the transformation to scale and shift the image's pixel values
    rescaled_image = image.astype(np.float32) * scale + shift

    # Clip the transformed values to ensure they remain within the specified range
    rescaled_image = np.clip(rescaled_image, min_val, max_val)

    return imagej_8B(rescaled_image)

def equalize(image: np.ndarray, min_val: float, max_val: float):
    
    clipped = np.clip(image, min_val, max_val)
    equalized = (clipped - min_val) / (max_val - min_val) * 255
    return equalized.astype(np.uint8)
