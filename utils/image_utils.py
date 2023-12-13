import numpy as np
import tifffile
import zarr
from typing import *
import cv2


def read_tif(path: str = None):
    """
    Read a TIFF file and return its contents as a NumPy array.

    Args:
        path (str): The path to the TIFF file.

    Returns:
        numpy.ndarray: The contents of the TIFF file as a NumPy array.
    """

    return tifffile.TiffFile(path).asarray()


def read_virtual_tif(path: str = None):
    """
    Read a virtual TIF file and return it as a zarr array.

    Args:
        path (str): The path to the virtual TIF file.

    Returns:
        zarr.core.Array: The zarr array representing the virtual TIF file.
    """
    return zarr.open(tifffile.TiffFile(path).aszarr())


def translate_image(image, dx, dy):
    """
    Translate the image by dx and dy.

    Args:
    image (np.ndarray): The image to be translated.
    dx (float): Translation along the x-axis.
    dy (float): Translation along the y-axis.

    Returns:
    np.ndarray: The translated image.
    """
    rows, cols = image.shape[:2]
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    return translated_image
