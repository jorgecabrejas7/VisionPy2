# utils/image_sequence.py

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tifffile
import zarr
from PyQt6.QtWidgets import *
from typing import List

from utils.progress_window import ProgressWindow

from pathlib import Path


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

def read_sequence2(folder_path,progress_window = None):
    """
    Read a sequence of TIFF files in a folder as a 3D volume.
    
    Args:
    folder_path (str): Path to the folder containing TIFF files.

    Returns:
    numpy.ndarray: A 3D array where each slice corresponds to a TIFF file.
    """

    # List and sort the TIFF files
    tiff_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if (f.endswith('.tiff') or f.endswith('.tif'))])

    tiff_sequence = tifffile.TiffSequence(tiff_files)
    
    # Get the total number of TIFF files
    total_files = len(tiff_files)
    
    # Read each TIFF file and update progress
    volume = []

    if progress_window == None:
    
        for i, file_path in enumerate(tiff_files):
            slice_data = tifffile.imread(file_path)
            volume.append(slice_data)
                
            # Update progress
    
    else:

        for i, file_path in enumerate(tiff_files):
            slice_data = tifffile.imread(file_path)
            volume.append(slice_data)
            progress_window.update_progress(int(i / total_files * 100), f"Loading: {os.path.basename(file_path)}",i,total_files)
            
        # Update progress
    
    return np.array(volume)

def read_virtual_sequence(folder_path: str, mode: str = "r", chunkmode: int = tifffile.CHUNKMODE.FILE) -> zarr.Array:
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
    volume_as_zarr = tiff_sequence.aszarr(chunkmode=chunkmode)

    return zarr.open(volume_as_zarr, mode=mode)

def write_sequence2(folder_path, name, volume, progress_window=None):
    """
    Save a 3D volume as a sequence of TIFF files in a folder.
    
    Args:
    folder_path (str): Path to the folder where TIFF files will be saved.
    name (str): Name of the TIFF files.
    volume (numpy.ndarray): A 3D array where each slice corresponds to an image.
    """

    folder_path = folder_path 

    # Create the folder if it doesn't exist
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    if progress_window == None:
            # Save each slice as a TIFF file
            for i in range(volume.shape[0]):
                tifffile.imwrite(f"{folder_path}/{name}_{i:04d}.tif", volume[i])
    else:
        total_files = volume.shape[0]
        for i in range(volume.shape[0]):
            tifffile.imwrite(f"{folder_path}/{name}_{i:04d}.tif", volume[i])
            progress_window.update_progress(int(i / total_files * 100), f"Saving: {name}_{i:04d}.tif",i,total_files)
    
    print("Saving complete.")