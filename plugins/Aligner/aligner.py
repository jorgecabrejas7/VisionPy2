import numpy as np
import cv2
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.measure import regionprops
from scipy.ndimage import rotate
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import binary_fill_holes
from skimage import feature
from joblib import Parallel, delayed
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation as R

def get_lines(image):

    # convert image type to uint8
    image = image.astype(np.uint8)

    # Perform Hough Line Transform
    lines = cv2.HoughLines(image, 1, np.pi / 180, 200)

    # Draw lines on the image
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # show image using matplotlib and creating a figure
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.show()

def get_lines2(image):
    """
    This function takes an image as input and draws a line from the first pixel to the last pixel of the image.
    
    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The output image with a line drawn from the first pixel to the last pixel of the input image.
    """
    print(image.shape)

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

def get_user_inputs3():
    """
    This function creates a dialog window with three checkboxes labeled "Main", "Left", and "Top", and an "Ok" button.
    The state of the checkboxes after the user closes the dialog is returned as a dictionary.

    Returns:
    dict: A dictionary with the state of the checkboxes. The keys are "Main", "Left", and "Top", and the values are boolean indicating whether the checkbox is checked or not.
    """
    # Create a window with 3 checkboxes and an Ok button
    window = QDialog()
    window.setWindowTitle("Select Reslices")
    window.setWindowModality(Qt.WindowModality.WindowModal)
    window.resize(300, 100)
    window.setLayout(QVBoxLayout())

    main_checkbox = QCheckBox("Main")
    left_checkbox = QCheckBox("Left")
    top_checkbox = QCheckBox("Top")

    ok_button = QPushButton("Ok")

    window.layout().addWidget(main_checkbox)
    window.layout().addWidget(left_checkbox)
    window.layout().addWidget(top_checkbox)
    window.layout().addWidget(ok_button)
    ok_button.clicked.connect(window.close)
    window.exec()

    main = main_checkbox.isChecked()
    left = left_checkbox.isChecked()
    top = top_checkbox.isChecked()

    return {"Main": main, "Left": left, "Top": top}

def get_angles(plugin, volume, reslices):
    """
    This function calculates the rotation angles for different slices of a 3D volume based on user-defined thresholds. 
    It uses Otsu's method to threshold the middle slice of the volume and identifies the largest connected component. 
    The orientation of this component is then used to calculate the rotation angle.

    Parameters:
    plugin (object): An instance of the plugin that is calling this function.
    volume (numpy.ndarray): A 3D numpy array representing the volume.
    reslices (dict): A dictionary indicating which slices to consider. Keys are "Main", "Left", and "Top", and values are boolean.

    Returns:
    dict: A dictionary containing the calculated rotation angles for the slices. The keys are "Main", "Left", and "Top", and the values are the calculated rotation angles in degrees.

    """

    # detect if the volume is 8bits or 16bits
    if volume.dtype == np.uint8:
        is8bit = True
    else:
        is8bit = False

    # names of the keys in the dictionary which value is True
    names = [key for key, value in reslices.items() if value == True]

    angles = {"Main": None, "Left": None, "Top": None}

    if not reslices["Main"]:
        if not reslices["Left"]:
            volume = np.transpose(volume, (2, 0, 1))
            volume = np.transpose(volume, (2, 0, 1))
        else:
            volume = np.transpose(volume, (2, 0, 1))

    if is8bit:
        top_threshold = 255
    else:
        top_threshold = 65535

    for name in names:
        while True:
            if name == "Top":
                if not reslices["Left"] and reslices["Main"]:
                    volume = np.transpose(volume, (2, 0, 1))

            top_threshold = get_user_inputs(plugin, name, default_value=top_threshold)

            # Check if the threshold is valid by aplying it to the middle slice
            if top_threshold != None:
                if name == "Top":
                    middle_slice = volume[len(volume) // 3].copy()
                else:
                    middle_slice = volume[len(volume) // 2].copy()

                if name == "Top":
                    # rotate the middle slice by 90 degree
                    middle_slice = np.rot90(middle_slice)

                # crop the middle slice to its half to avoid the background
                middle_slice = middle_slice[
                    middle_slice.shape[0] // 4 : middle_slice.shape[0] // 4 * 3, :
                ]

                top_index = np.where(middle_slice > top_threshold)

                middle_slice[top_index] = 0

                if is8bit:
                    threshold_value = threshold_otsu(
                        middle_slice[middle_slice > 10]
                    )
                else:
                    threshold_value = threshold_otsu(
                        middle_slice[middle_slice > 10000]
                    )

                print("threshold value is: ", threshold_value)

                thresholded_slice = middle_slice > threshold_value

                thresholded_slice_to_print = thresholded_slice.copy()

                # Label the objects in the thresholded slice
                labeled_slice = label(thresholded_slice)

                # Get the properties of each labeled region
                regions = regionprops(labeled_slice)

                if regions == []:
                    # Show a message box if invalid threshold is selected
                    msg_box = QMessageBox(plugin.main_window)
                    msg_box.setText("To low threshold selected.")
                    msg_box.exec()
                    continue

                # Find the largest connected component
                largest_component = max(regions, key=lambda region: region.area)

                # Create a mask to keep only the largest component
                mask = np.zeros_like(labeled_slice)
                mask[labeled_slice == largest_component.label] = 1
                mask = binary_fill_holes(mask).astype(int)

                # Apply the mask to the thresholded slice
                thresholded_slice = thresholded_slice * mask

                # extract edges using canny edge detector
                mask = feature.canny(mask > 0, sigma=0) > 0

                # Label the objects in the thresholded slice
                mask = label(mask)
                # print np unique of mask
                print(np.unique(mask))

                # Get the properties of each labeled region
                regions = regionprops(mask)

                # Find the second largest connected component if there is one, if not the first
                if len(regions) == 1:
                    second_largest_component = regions[0]
                else:
                    second_largest_component = sorted(
                        regions, key=lambda region: region.area
                    )[-1]

                # delete everything that is not the second largest component from mask
                mask[mask != second_largest_component.label] = 0

                try:
                    mask = get_lines2(mask)
                except:
                    print("line not smoothed")

                # Compute the rotation angle of the largest component
                rotation_angle = second_largest_component.orientation

                # Convert the angle from radians to degrees
                rotation_angle_degrees = np.degrees(rotation_angle)

                rotated_slice = rotate(thresholded_slice, -rotation_angle_degrees)

                # Print the rotation angle
                print(
                    f"The rotation angle of the largest component is {rotation_angle_degrees} degrees."
                )

                plt.figure()
                # Plotting the original middle slice
                plt.subplot(2, 2, 1)
                plt.imshow(middle_slice, cmap="gray")
                plt.title("Original Middle Slice")

                # Plotting the thresholded slice
                plt.subplot(2, 2, 2)
                plt.imshow(thresholded_slice_to_print, cmap="gray")
                plt.title("Thresholded Slice")

                # Plotting the largest component mask
                plt.subplot(2, 2, 3)
                plt.imshow(mask, cmap="gray")
                plt.title("Largest Component Mask")

                # Plotting the rotated slice
                plt.subplot(2, 2, 4)
                plt.imshow(rotated_slice, cmap="gray")
                plt.title("Rotated Slice")

                # Adjusting the layout and displaying the plot
                plt.tight_layout()
                plt.show()

                plt.show()

                # Ask the user if the threshold is ok
                msg_box = QMessageBox(plugin.main_window)
                msg_box.setText(f"Is the threshold ok for {name} reslice?")
                msg_box.setStandardButtons(
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)
                ret = msg_box.exec()

                # close the figure
                plt.close()

                if ret == QMessageBox.StandardButton.Yes:
                    angles[name] = -rotation_angle_degrees
                    volume = np.transpose(volume, (2, 0, 1))
                    break
                else:
                    continue
            else:
                # Show a message box if no threshold is selected
                msg_box = QMessageBox(plugin.main_window)
                msg_box.setText("No threshold selected.")
                msg_box.exec()

    return angles

def get_angles_auto(plugin, volume, reslices): #get the angles without asking threshold
    """
    This function calculates the rotation angles for different slices of a 3D volume based on user-defined thresholds. 
    It uses Otsu's method to threshold the middle slice of the volume and identifies the largest connected component. 
    The orientation of this component is then used to calculate the rotation angle.

    Parameters:
    plugin (object): An instance of the plugin that is calling this function.
    volume (numpy.ndarray): A 3D numpy array representing the volume.
    reslices (dict): A dictionary indicating which slices to consider. Keys are "Main", "Left", and "Top", and values are boolean.

    Returns:
    dict: A dictionary containing the calculated rotation angles for the slices. The keys are "Main", "Left", and "Top", and the values are the calculated rotation angles in degrees.

    """

    # detect if the volume is 8bits or 16bits
    if volume.dtype == np.uint8:
        is8bit = True
    else:
        is8bit = False

    # names of the keys in the dictionary which value is True
    names = [key for key, value in reslices.items() if value == True]

    angles = {"Main": None, "Left": None, "Top": None}

    if not reslices["Main"]:
        if not reslices["Left"]:
            volume = np.transpose(volume, (2, 0, 1))
            volume = np.transpose(volume, (2, 0, 1))
        else:
            volume = np.transpose(volume, (2, 0, 1))

    if is8bit:
        top_threshold = 255
    else:
        top_threshold = 65535

    for name in names:
        if name == "Top":
            if not reslices["Left"] and reslices["Main"]:
                volume = np.transpose(volume, (2, 0, 1))

        top_threshold = 255
        
        # Check if the threshold is valid by aplying it to the middle slice
        if top_threshold != None:
            if name == "Top":
                middle_slice = volume[len(volume) // 3].copy()
            else:
                middle_slice = volume[len(volume) // 2].copy()

            if name == "Top":
                # rotate the middle slice by 90 degree
                middle_slice = np.rot90(middle_slice)

            # crop the middle slice to its half to avoid the background
            middle_slice = middle_slice[
                middle_slice.shape[0] // 4 : middle_slice.shape[0] // 4 * 3, :
            ]

            top_index = np.where(middle_slice > top_threshold)

            middle_slice[top_index] = 0

            if is8bit:
                threshold_value = threshold_otsu(
                    middle_slice[middle_slice > 10]
                )
            else:
                threshold_value = threshold_otsu(
                    middle_slice[middle_slice > 10000]
                )

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

            # extract edges using canny edge detector
            mask = feature.canny(mask > 0, sigma=0) > 0

            # Label the objects in the thresholded slice
            mask = label(mask)
            # print np unique of mask
            print(np.unique(mask))

            # Get the properties of each labeled region
            regions = regionprops(mask)

            # Find the second largest connected component if there is one, if not the first
            if len(regions) == 1:
                second_largest_component = regions[0]
            else:
                second_largest_component = sorted(
                    regions, key=lambda region: region.area
                )[-1]

            # delete everything that is not the second largest component from mask
            mask[mask != second_largest_component.label] = 0

            try:
                mask = get_lines2(mask)
            except:
                print("line not smoothed")

            # Compute the rotation angle of the largest component
            rotation_angle = second_largest_component.orientation

            # Convert the angle from radians to degrees
            rotation_angle_degrees = np.degrees(rotation_angle)

            # Print the rotation angle
            print(
                f"The rotation angle of the largest component is {rotation_angle_degrees} degrees."
            )

            angles[name] = -rotation_angle_degrees
            volume = np.transpose(volume, (2, 0, 1))

        else:
            # Show a message box if no threshold is selected
            msg_box = QMessageBox(plugin.main_window)
            msg_box.setText("No threshold selected.")
            msg_box.exec()

    return angles

def get_user_inputs(plugin, name, default_value=65535):
    """
    This function creates a dialog window that prompts the user to enter a threshold value for a given reslice name. 
    The dialog window is created in the main window of the plugin.

    Parameters:
    plugin (object): An instance of the plugin that is calling this function.
    name (str): The name of the reslice for which the threshold is being set.
    default_value (int, optional): The default threshold value. Defaults to 65535.

    Returns:
    int or None: The threshold value if the user clicks "Ok", None otherwise.
    """
    # Get the threshold value for the name reslice
    threshold, ok = QInputDialog.getInt(
        plugin.main_window,
        "Threshold",
        f"Enter Threshold for {name} reslice:",
        default_value,
    )

    if ok:
        return threshold
    else:
        return None

def rotate_volume( volume, angle, progress_window=None):
    """
    This function rotates a 3D volume by a given angle. The rotation is applied to each slice of the volume. 
    If a progress window is provided, the progress of the rotation is displayed in it.

    Parameters:
    volume (numpy.ndarray): A 3D array where each slice corresponds to an image.
    angle (float): The angle in degrees by which to rotate the volume.
    progress_window (object, optional): An instance of a progress window in which to display the progress of the rotation. Defaults to None.

    Returns:
    numpy.ndarray: The rotated 3D volume.
    """

    rotated = rotate(volume[0], angle)

    shape = shape = (volume.shape[0],) + rotated.shape

    rotated_volume = np.zeros(shape=shape, dtype=volume.dtype)

    print(shape)

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

def rotate_volume_concurrent(volume, angle, progress_window=None):
    """
    This function rotates a 3D volume by a given angle using concurrent processing with joblib. 
    The rotation is applied to each slice of the volume concurrently. 
    If a progress window is provided, the progress of the rotation is displayed in it.

    Parameters:
    volume (numpy.ndarray): A 3D array where each slice corresponds to an image.
    angle (float): The angle in degrees by which to rotate the volume.
    progress_window (object, optional): An instance of a progress window in which to display the progress of the rotation. Defaults to None.

    Returns:
    numpy.ndarray: The rotated 3D volume.
    """
    def rotate_slice(slice):
        return rotate(slice, angle)

    rotated = rotate(volume[0], angle)
    shape = (volume.shape[0],) + rotated.shape
    rotated_volume = np.zeros(shape=shape, dtype=volume.dtype)

    if progress_window is None:
        rotated_slices = Parallel(n_jobs=-1)(delayed(rotate_slice)(volume[i]) for i in range(volume.shape[0]))
        for i, result in enumerate(rotated_slices):
            rotated_volume[i] = result
    else:
        rotated_slices = Parallel(n_jobs=-1)(delayed(rotate_slice)(volume[i]) for i in range(volume.shape[0]))
        for i, result in enumerate(rotated_slices):
            rotated_volume[i] = result
            progress_window.update_progress(
                int(i / volume.shape[0] * 100),
                f"Rotating: {i}",
                i,
                volume.shape[0],
            )

    return rotated_volume

def align_sample(volume, mask):
    """
    Aligns the CFRP sample with the coordinate axes.

    Parameters:
    volume (np.ndarray): 3D NumPy array of the grayscale volume.
    mask (np.ndarray): 3D NumPy array of the binary mask of what is going to be aligned.

    Returns:
    aligned_volume (np.ndarray): Aligned grayscale volume.
    aligned_mask (np.ndarray): Aligned binary mask.
    """
    #check if mask and volume are the same size
    if volume.shape != mask.shape:
        raise ValueError("Volume and mask must have the same shape.")
    # Estimate the orientation of the sample using the binary mask
    coords = np.column_stack(np.where(mask))
    cov_matrix = np.cov(coords, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    # Ensure the eigenvectors are sorted by eigenvalues in descending order
    sorted_indices = np.argsort(-eigvals)
    eigvecs = eigvecs[:, sorted_indices]

    # Compute the best transformation (rotation) to align the sample with the axes
    rotation_matrix = eigvecs
    rotation = R.from_matrix(rotation_matrix)
    rotation_matrix = rotation.as_matrix()

    # Center of the volume
    center = np.array(volume.shape) / 2

    # Apply the transformation to both the volume and the mask
    aligned_volume = affine_transform(volume, rotation_matrix, offset=center - rotation_matrix @ center, order=1)
    aligned_mask = affine_transform(mask, rotation_matrix, offset=center - rotation_matrix @ center, order=0)

    return aligned_volume, aligned_mask
