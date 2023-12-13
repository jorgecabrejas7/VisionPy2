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
    