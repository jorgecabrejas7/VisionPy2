from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.measure import regionprops
from scipy.ndimage import binary_fill_holes
from skimage import feature
from scipy.ndimage import rotate
import scipy


def get_angle(volume, gate):
    """
    Get the threshold values for different slices of a volume.

    Args:
    - self: the object instance
    - volume: a 3D numpy array representing the volume

    Returns:
    - user_inputs: a dictionary containing the threshold values for different slices

    Raises:
    - No specific exceptions are raised within this function.

    Example:
    ```python
    # Create an instance of the class
    instance = ClassName()

    # Define a 3D numpy array representing the volume
    volume = np.random.rand(10, 10, 10)

    # Call the function to get the threshold values
    thresholds = instance.get_angles(volume)
    ```
    """

    sliceid = gate[0]

    middle_slice = volume[sliceid]

    # otsu threshold
    threshold_value = threshold_otsu(middle_slice)
    print("threshold value is: ", threshold_value)
    thresholded_slice = middle_slice > threshold_value

    # Label the objects in the thresholded slice
    labeled_slice = label(thresholded_slice)

    # Get the properties of each labeled region
    regions = regionprops(labeled_slice)

    # Find the largest connected component
    largest_component = max(regions, key=lambda region: region.area)

    # Create a mask to keep only the largest component
    mask = np.zeros_like(labeled_slice)
    mask[labeled_slice == largest_component.label] = 1
    mask = binary_fill_holes(mask).astype(int)

    # Apply the mask to the thresholded slice
    thresholded_slice = thresholded_slice * mask

    # fill holes in the mask
    mask = binary_fill_holes(mask).astype(int)

    # extract edges using canny edge detector
    mask = feature.canny(mask > 0, sigma=0) > 0

    # Label the objects in the thresholded slice
    mask = label(mask)

    # Get the properties of each labeled region
    regions = regionprops(mask)

    # Find the largest connected component if there is one, if not the first
    if len(regions) == 1:
        largest_component = regions[0]
    else:
        largest_component = sorted(regions, key=lambda region: region.area)[-1]

    # delete everything that is not the largest component from mask
    mask[mask != largest_component.label] = 0
    try:
        mask = get_lines2(mask)
    except:
        print("line not smoothed")

    # Compute the rotation angle of the largest component
    rotation_angle = largest_component.orientation

    # Convert the angle from radians to degrees
    rotation_angle_degrees = np.degrees(rotation_angle)

    # Print the rotation angle
    print(
        f"The rotation angle of the largest component is {rotation_angle_degrees} degrees."
    )

    return -rotation_angle_degrees


def get_lines2(image):
    # get the first and last pixel of the image
    linea = np.where(image >= 1)
    x0 = linea[0][0]
    x1 = linea[0][-1]
    y0 = linea[1][0]
    y1 = linea[1][-1]

    # create a image with the same size as the input image
    image2 = np.zeros_like(image)

    import cv2

    # draw the line on the image2
    cv2.line(image2, (y0, x0), (y1, x1), (1, 1, 1), 2)

    return image2


def rotate_volume_deprecated(volume, angle, progress_window=None):
    """
    Rotate a 3D volume by a given angle.

    Args:
    volume (numpy.ndarray): A 3D array where each slice corresponds to an image.
    angle (float): Angle in degrees.

    Returns:
    numpy.ndarray: A rotated 3D array.
    """

    rotated = rotate(volume[0], angle)

    shape = shape = (volume.shape[0],) + rotated.shape

    rotated_volume = np.zeros(shape=shape, dtype=volume.dtype)

    if progress_window == None:
        for i in range(volume.shape[0]):
            rotated_volume[i] = rotate(volume[i], angle)
    # Rotate each slice of the volume with progress bar
    else:
        for i in range(volume.shape[0]):
            rotated_volume[i] = rotate(volume[i], angle)
            progress_window.update_progress(
                int(i / volume.shape[0] * 100), f"Rotating: {i}", i, volume.shape[0]
            )

    return rotated_volume


def rotate_volume(volume, angle, progress_window=None):
    volume = np.swapaxes(volume, 0, 1)
    volume = np.swapaxes(volume, 1, 2)

    theta = np.radians(angle)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array(
        [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
    )

    # Apply the affine transform
    rotated_volume = scipy.ndimage.affine_transform(
        volume, rotation_matrix, order=1, mode="constant", cval=128
    )

    rotated_volume = np.swapaxes(rotated_volume, 1, 2)
    rotated_volume = np.swapaxes(rotated_volume, 0, 1)

    return rotated_volume


def ask_gate(plugin):
    # create a window
    start, ok = QInputDialog.getInt(
        plugin.main_window, "Start", "Enter start of the gate", 0
    )
    end, ok = QInputDialog.getInt(
        plugin.main_window, "End", "Enter start of the gate", 0
    )

    if ok:
        return (start, end)


def plot_signal_gate(signal, gate):
    # clean the plot
    plt.close("all")
    plt.plot(signal)
    plt.axvline(x=gate[0], color="r", linestyle="--")
    plt.axvline(x=gate[1], color="r", linestyle="--")
    plt.show()


def get_gate(plugin, data):
    max_signal = np.max(data, axis=(1, 2))

    plugin.request_gui(plt.plot, max_signal)

    while True:
        while True:
            gate = plugin.request_gui(ask_gate, plugin)

            # gate end must be greater than gate start and start must be greater than 0
            if (gate[0] < gate[1]) and (gate[0] > 0):
                break

        # plot the signal and the gate

        plugin.request_gui(plot_signal_gate, max_signal, gate)

        # ask the user if the gate is correct
        response = plugin.prompt_confirmation("Is the gate correct?")

        if response:
            return gate


def align(data, gate):
    # now do it for the whole volume
    rolled_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            signal = data[:, i, j]
            gated_data = signal[gate[0] : gate[1]]
            max_gated_data_index = np.argmax(gated_data, axis=0)
            rolled = np.roll(signal, -max_gated_data_index)
            rolled_data[:, i, j] = rolled
    return rolled_data


def autogate(volume):
    brightest_slice = np.argmax(np.sum(volume, axis=(1, 2)))

    return [brightest_slice - 2, brightest_slice + 2]


def ask_auto(plugin):
    return plugin.prompt_confirmation("Do you want to use the auto gate?")
