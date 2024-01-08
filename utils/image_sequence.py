# utils/image_sequence.py

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tifffile
import zarr
from PyQt6.QtWidgets import *
from typing import List

from utils.progress_window import ProgressWindow


def read_sequence(folder_path):
    """
    Read a sequence of TIFF files in a folder as a 3D volume.

    Args:
        folder_path (str): Path to the folder containing TIFF files.

    Returns:
        numpy.ndarray: A 3D array where each slice corresponds to a TIFF file.

    Raises:
        FileNotFoundError: If the folder_path does not exist or is inaccessible.
        ValueError: If no TIFF files are found in the folder.

    Example:
        >>> folder_path = '/path/to/tiff/folder'
        >>> volume = read_sequence(folder_path)
    """

    # List and sort the TIFF files
    tiff_files = sorted(
        [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".tif") or f.endswith(".tiff")
        ]
    )

    tiff_sequence = tifffile.TiffSequence(tiff_files)

    return tiff_sequence.asarray(ioworkers=5)


def read_virtual_sequence(folder_path: str, mode: str = "r") -> zarr.Array:
    """
    Read a sequence of TIFF files in a folder as a 3D volume.

    Args:
        folder_path (str): Path to the folder containing TIFF files.

    Returns:
        zarr.Array: A 3D array where each slice corresponds to a TIFF file.

    Raises:
        FileNotFoundError: If the specified folder_path does not exist.
        ValueError: If no TIFF files are found in the specified folder.

    """

    # List and sort the TIFF files in the specified folder
    # Each file is expected to be a slice in the 3D volume
    tiff_files: List[str] = sorted(
        [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".tif") or f.endswith(".tiff")
        ]
    )

    # Create a TiffSequence object from the list of TIFF files
    # TiffSequence represents a sequence of TIFF files as a single entity
    tiff_sequence = tifffile.TiffSequence(tiff_files)

    # Convert the TiffSequence to a Zarr array
    # This enables lazy-loading of image slices, reducing memory usage for large datasets
    volume_as_zarr = tiff_sequence.aszarr()

    return zarr.open(volume_as_zarr, mode=mode)
