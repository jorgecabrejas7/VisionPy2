import tifffile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
from skimage import measure
from skimage.measure import regionprops
from skimage import filters
import fill_voids

def onlypores(xct):

    print('masking')

    # Create a masked array, excluding zeros
    masked_data = np.ma.masked_equal(xct, 0)

    unmasked_data = masked_data.compressed()

    print('computing otsu')

    # Apply Otsu thresholding on the non-zero values
    threshold_value = filters.threshold_otsu(unmasked_data)

    print('thresholding with value: ', threshold_value)

    binary = xct > threshold_value

    max_proj = np.max(binary, axis=0)

    labels = measure.label(max_proj)

    props = regionprops(labels)

    minr, minc, maxr, maxc = props[0].bbox

    #crop the volume

    binary_cropped = binary[:, minr:maxr, minc:maxc]

    sample_mask_cropped = fill_voids.fill(binary_cropped, in_place=False)

    sample_mask = np.zeros_like(binary)
    sample_mask[:, minr:maxr, minc:maxc] = sample_mask_cropped

    #invert binary
    binary_inverted = np.invert(binary)
    onlypores = np.logical_and(binary_inverted, sample_mask)

    return onlypores, sample_mask, binary