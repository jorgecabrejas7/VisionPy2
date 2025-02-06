import numpy as np
from skimage import measure
from skimage.measure import regionprops
from skimage import filters
import fill_voids


def onlypores(xct):
    print("masking")

    # Create a masked array, excluding zeros
    masked_data = np.ma.masked_equal(xct, 0)

    unmasked_data = masked_data.compressed()

    print("computing otsu")

    # Apply Otsu thresholding on the non-zero values
    threshold_value = filters.threshold_otsu(unmasked_data)

    print("thresholding with value: ", threshold_value)

    binary = xct > threshold_value

    max_proj = np.max(binary, axis=0)

    labels = measure.label(max_proj)

    props = regionprops(labels)

    minr, minc, maxr, maxc = props[0].bbox

    # crop the volume

    binary_cropped = binary[:, minr:maxr, minc:maxc]

    sample_mask_cropped = fill_voids.fill(binary_cropped, in_place=False)

    sample_mask = np.zeros_like(binary)
    sample_mask[:, minr:maxr, minc:maxc] = sample_mask_cropped

    # invert binary
    binary_inverted = np.invert(binary)
    onlypores = np.logical_and(binary_inverted, sample_mask)

    return onlypores, sample_mask, binary


def onlypores_parallel(xct):
    from joblib import Parallel, delayed
    import numpy as np

    # Function to apply the mask and compress the data
    def mask_and_compress(xct_chunk):
        masked_data = np.ma.masked_equal(xct_chunk, 0)
        return masked_data.compressed()

    print("masking")

    # Number of chunks (adjust depending on the size of your array and available cores)
    num_chunks = 16  # You can increase or decrease this based on testing

    # Assuming xct is your large array, divide it into chunks
    chunks = np.array_split(xct, num_chunks)

    # Use joblib to apply the function in parallel
    compressed_chunks = Parallel(n_jobs=-1, backend="loky")(
        delayed(mask_and_compress)(chunk) for chunk in chunks
    )

    # Combine the results back into a single array
    unmasked_data = np.concatenate(compressed_chunks)

    print("computing otsu")

    # Apply Otsu thresholding on the non-zero values
    threshold_value = filters.threshold_otsu(unmasked_data)

    print("thresholding with value: ", threshold_value)

    binary = xct > threshold_value

    max_proj = np.max(binary, axis=0)

    labels = measure.label(max_proj)

    props = regionprops(labels)

    minr, minc, maxr, maxc = props[0].bbox

    # crop the volume

    binary_cropped = binary[:, minr:maxr, minc:maxc]

    sample_mask_cropped = fill_voids.fill(binary_cropped, in_place=False)

    sample_mask = np.zeros_like(binary)
    sample_mask[:, minr:maxr, minc:maxc] = sample_mask_cropped

    # invert binary
    binary_inverted = np.invert(binary)
    onlypores = np.logical_and(binary_inverted, sample_mask)

    return onlypores, sample_mask, binary
