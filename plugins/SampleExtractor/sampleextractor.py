import numpy as np
from tqdm import tqdm
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes
from joblib import Parallel, delayed
from skimage.measure import label, regionprops


def process_volume(volume, n_samples):
    """
    Process the volume to extract sample volumes.

    Args:
    volume (numpy.ndarray): The 3D volume to process.
    n_samples (int): The number of samples to extract.

    Returns:
    list: A list of sample volumes.
    """
    # Otsu threshold
    thresh = threshold_otsu(volume)
    binary = volume > thresh

    # Binary fill holes
    filled = np.zeros_like(binary)

    def process_slice(i):
        return binary_fill_holes(binary[i])

    # Parallel processing
    filled_slices = Parallel(n_jobs=-1)(
        delayed(process_slice)(i) for i in tqdm(range(binary.shape[0]))
    )

    # Combine the results
    for i in range(binary.shape[0]):
        filled[i] = filled_slices[i]

    # Binary dilate
    label_image = label(filled)

    props = regionprops(label_image)

    # Sort props by area
    props.sort(key=lambda x: x.area, reverse=True)

    def process_sample(volume, label_image, label, bbox):
        sample = volume.copy()
        value = sample[sample.shape[0] // 2, sample.shape[1] // 2].min()
        sample[label_image != label] = value
        sample = sample[bbox[0] : bbox[3], bbox[1] : bbox[4], bbox[2] : bbox[5]]
        return sample

    volumes = Parallel(n_jobs=-1)(
        delayed(process_sample)(volume, label_image, props[i].label, props[i].bbox)
        for i in tqdm(range(n_samples))
    )

    bboxes = [props[i].bbox for i in range(n_samples)]

    # order the samples by bbox[2]

    volumes = [v for _, v in sorted(zip(bboxes, volumes), key=lambda pair: pair[0][2])]

    return volumes
