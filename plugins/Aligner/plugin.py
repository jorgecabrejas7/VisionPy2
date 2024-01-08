# Import necessary modules
from base_plugin import BasePlugin
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from utils.image_sequence import read_sequence
from utils.progress_window import ProgressWindow
from pathlib import Path
import traceback
import sys
import os
import plugins.CScan.NDEToolkit as ndt
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.measure import regionprops
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


class Plugin(BasePlugin):
    def __init__(self, main_window, plugin_name):
        super().__init__(main_window, plugin_name)

    def execute(self):
        """
        Prompt the user to select a file and process the selected file.

        Raises:
            FileNotFoundError: If no file is selected.

        Example:
            execute()
        """
        file_path = self.select_folder(caption="Select Folder")
        if not file_path:
            self.prompt_error("No file selected")
            self.finished.emit(self)
            return

        save_path = self.select_folder(caption="Select Folder to save rotated volume")
        if not save_path:
            self.prompt_error("No file selected")
            self.finished.emit(self)
            return

        # Read volume
        volume = read_sequence(file_path)
        user_inputs = self.requesr_gui(self.get_thresholds, volume)
        if not user_inputs:
            self.prompt_error("No file selected")
            self.finished.emit(self)
            return

        # Rotate volume
        angle = self.get_angle(volume, show=True, top_threshold=user_inputs["Main"])
        volume = self.rotate_volume_concurrent(
            volume,
            angle,
            progress_window=True,
            message="Rotating Volume from Main reslice",
        )

        volume = np.transpose(volume, (2, 0, 1))
        angle = self.get_angle(volume, show=True, top_threshold=user_inputs["Left"])
        volume = self.rotate_volume_concurrent(
            volume,
            angle,
            progress_window=True,
            message="Rotating Volume from Left reslice",
        )

        volume = np.transpose(volume, (2, 0, 1))
        angle = self.get_angle(
            volume, show=True, top_threshold=user_inputs["Top"], rotate90=True
        )
        volume = self.rotate_volume_concurrent(
            volume,
            angle,
            progress_window=True,
            message="Rotating Volume from Top reslice",
        )

        volume = np.transpose(volume, (2, 0, 1))

    """
    Plugin functions
    """

    def get_angle(self, volume, show=False, top_threshold=65535, rotate90=False):
        middle_slice = volume[len(volume) // 2]

        if rotate90:
            middle_slice = np.rot90(middle_slice)

        top_index = np.where(middle_slice > top_threshold)

        threshold_value = threshold_otsu(middle_slice[middle_slice > 10000])

        print("threshold value is: ", threshold_value)

        thresholded_slice = middle_slice > threshold_value

        thresholded_slice[top_index] = 0

        # Label the objects in the thresholded slice
        labeled_slice = label(thresholded_slice)

        # Get the properties of each labeled region
        regions = regionprops(labeled_slice)

        # Find the largest connected component
        largest_component = max(regions, key=lambda region: region.area)

        # Create a mask to keep only the largest component
        mask = np.zeros_like(labeled_slice)
        mask[labeled_slice == largest_component.label] = 1

        # Apply the mask to the thresholded slice
        thresholded_slice = thresholded_slice * mask

        # Compute the rotation angle of the largest component
        rotation_angle = largest_component.orientation

        # Convert the angle from radians to degrees
        rotation_angle_degrees = np.degrees(rotation_angle)

        rotated_slice = rotate(thresholded_slice, -rotation_angle_degrees)

        def plot_angles():
            plt.figure()

            plt.hist(middle_slice[middle_slice > 10000].ravel(), bins=256)
            plt.title("Histogram")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            plt.axvline(threshold_value, color="r", linestyle="dashed", linewidth=2)
            plt.show()

            plt.figure()
            if show:
                # Print the rotation angle
                print(
                    f"The rotation angle of the largest component is {rotation_angle_degrees} degrees."
                )

                # Plotting the original middle slice
                plt.subplot(2, 2, 1)
                plt.imshow(middle_slice, cmap="gray")
                plt.title("Original Middle Slice")

                # Plotting the thresholded slice
                plt.subplot(2, 2, 2)
                plt.imshow(thresholded_slice, cmap="gray")
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

        self.request_gui(plot_angles)

        return -rotation_angle_degrees

    def get_thresholds(self, volume):
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
        thresholds = instance.get_thresholds(volume)
        ```
        """

        names = ["Main", "Left", "Top"]

        user_inputs = {"Main": None, "Left": None, "Top": None}

        top_threshold = 65535

        for name in names:
            while True:
                top_threshold = self.request_gui(
                    self.get_user_inputs, name, default_value=top_threshold
                )

                # Check if the threshold is valid by aplying it to the middle slice
                if top_threshold is not None:
                    middle_slice = volume[len(volume) // 2]

                    top_index = np.where(middle_slice > top_threshold)

                    threshold_value = threshold_otsu(middle_slice[middle_slice > 10000])

                    print("threshold value is: ", threshold_value)

                    thresholded_slice = middle_slice > threshold_value

                    thresholded_slice[top_index] = 0

                    # Label the objects in the thresholded slice
                    labeled_slice = label(thresholded_slice)

                    # Get the properties of each labeled region
                    regions = regionprops(labeled_slice)

                    if regions == []:
                        # Show a message box if invalid threshold is selected
                        self.prompt_error("To low threshold selected.")
                        continue

                    # Find the largest connected component
                    largest_component = max(regions, key=lambda region: region.area)

                    # Create a mask to keep only the largest component
                    mask = np.zeros_like(labeled_slice)
                    mask[labeled_slice == largest_component.label] = 1

                    # Apply the mask to the thresholded slice
                    thresholded_slice = thresholded_slice * mask

                    # Compute the rotation angle of the largest component
                    rotation_angle = largest_component.orientation

                    # Convert the angle from radians to degrees
                    rotation_angle_degrees = np.degrees(rotation_angle)

                    rotated_slice = rotate(thresholded_slice, -rotation_angle_degrees)

                    # Print the rotation angle
                    print(
                        f"The rotation angle of the largest component is {rotation_angle_degrees} degrees."
                    )

                    def threshold_ok():
                        # Plotting the original middle slice
                        plt.subplot(2, 2, 1)
                        plt.imshow(middle_slice, cmap="gray")
                        plt.title("Original Middle Slice")

                        # Plotting the thresholded slice
                        plt.subplot(2, 2, 2)
                        plt.imshow(thresholded_slice, cmap="gray")
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

                        msg_box = QMessageBox(self.parent)
                        msg_box.setText(f"Is the threshold ok for {name} reslice?")
                        msg_box.setStandardButtons(
                            QMessageBox.StandardButton.Yes
                            | QMessageBox.StandardButton.No
                        )
                        msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)
                        ret = msg_box.exec()
                        plt.close()

                        return ret == QMessageBox.StandardButton.Yes

                    # close the figure

                    ret = self.request_gui(threshold_ok)
                    if ret:
                        user_inputs[name] = top_threshold
                        volume = np.transpose(volume, (2, 0, 1))
                        break
                    else:
                        continue
                else:
                    # Show a message box if no threshold is selected
                    self.prompt_error("No threshold selected.")

        return user_inputs

    def get_user_inputs(self, name, default_value=65535):
        # Get the threshold value for the name reslice
        threshold, ok = QInputDialog.getInt(
            self.parent,
            "Threshold",
            f"Enter Threshold for {name} reslice:",
            default_value,
        )

        if ok:
            return threshold
        else:
            return None

    def rotate_volume(self, volume, angle, progress_window=None):
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

    def rotate_volume_concurrent(
        self, volume, angle, progress_window=False, message=None
    ):
        def rotate_slice(args):
            slice, angle = args
            return rotate(slice, angle)

        rotated = rotate(volume[0], angle)
        shape = (volume.shape[0],) + rotated.shape
        rotated_volume = np.zeros(shape=shape, dtype=volume.dtype)

        if not progress_window:
            with ThreadPoolExecutor() as executor:
                args = [(volume[i], angle) for i in range(volume.shape[0])]
                for i, result in enumerate(executor.map(rotate_slice, args)):
                    rotated_volume[i] = result
        else:
            progress_message = (
                message if message is not None else f"Rotating: {0}",
                0,
                volume.shape[0],
            )
            self.update_progress(0, progress_message)
            with ThreadPoolExecutor() as executor:
                args = [(volume[i], angle) for i in range(volume.shape[0])]

                for i, result in enumerate(executor.map(rotate_slice, args)):
                    rotated_volume[i] = result
                    self.update_progress(
                        int(i / volume.shape[0] * 100),
                        f"Rotating: {i}",
                        i,
                        volume.shape[0],
                    )

        return rotated_volume
