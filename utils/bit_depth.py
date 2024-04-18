import numpy as np


def f32_to_uint16(float_image: np.ndarray, do_scaling: bool = True) -> np.ndarray:
    """
    Converts a floating-point image to a short (16-bit) image.

    Args:
        float_image (ndarray): The input floating-point image.
        do_scaling (bool, optional): Whether to perform scaling. Defaults to True.

    Returns:
        ndarray: The converted short image.

    Raises:
        None

    Example:
        >>> input_image = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        >>> f32_to_uint16(input_image)
        array([[   0, 21845],
               [43690, 65535]], dtype=uint16)
    """

    min_val: float = float_image.min()
    max_val: float = float_image.max()
    scale: float = 65535.0 / (max_val - min_val) if max_val != min_val else 1.0

    if do_scaling:
        short_image: np.ndarray = (float_image - min_val) * scale
    else:
        short_image: np.ndarray = float_image

    short_image = np.clip(short_image, 0, 65535)
    short_image = np.round(short_image).astype(np.uint16)

    return short_image


def convert_to_8bit(image, use_scaling=False, min_val=None, max_val=None):
    """
    Converts an image to 8-bit. If use_scaling is True, it scales from the min-max range to 0-255.
    If use_scaling is False, it directly converts from 16-bit to 8-bit.

    Args:
        image (np.ndarray): The input image.
        use_scaling (bool): Determines whether to use scaling based on min_val and max_val.
        min_val (float, optional): Minimum value for scaling. If None, image's min is used.
        max_val (float, optional): Maximum value for scaling. If None, image's max is used.

    Returns:
        np.ndarray: The converted 8-bit image.
    """
    if use_scaling:
        # If min_val or max_val are not provided, use the image's min and max
        if min_val is None:
            min_val = np.min(image)
        if max_val is None:
            max_val = np.max(image)

        # Avoid division by zero
        if min_val == max_val:
            return np.zeros(image.shape, dtype=np.uint8)

        # Scale and convert to 8-bit
        scaled_image = (image - min_val) / (max_val - min_val) * 255
        return np.clip(scaled_image, 0, 255).astype(np.uint8)
    else:
        # Directly convert from 16-bit to 8-bit
        return (image / 65535.0 * 255).astype(np.uint8)

def imagej_8B(image: np.ndarray):
    
    min_val, max_val = image.min(), image.max()
    bins = np.linspace(min_val, max_val, 256) # 256 bins +1 to cover end point

    # Use digitize to find out indices of the bins each pixel belongs to
    indices = np.digitize(image, bins) - 1  # Subtract 1 to make indices 0-based

    # Since indices are in range 1 to 256, we subtract 1 to map to 0-255
    # Clip any out-of-bound values just in case
    img_8bit = np.clip(indices, 0, 255).astype(np.uint8)

    return img_8bit