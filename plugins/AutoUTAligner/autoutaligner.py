import numpy as np

def find_brightest_ut(volume):
    ut = volume.copy()

    ut = np.swapaxes(ut, 0, 1)
    ut = np.swapaxes(ut, 1, 2)

    # Find the brightest slice
    brightest_slice = np.argmax(np.sum(ut, axis=(0, 1)))

    return brightest_slice

def auto_gate(volume):
    # Find the brightest slice
    brightest_slice = find_brightest_ut(volume)

    return (brightest_slice - 2, brightest_slice + 2)