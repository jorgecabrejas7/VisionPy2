# Import necessary modules
from base_plugin import BasePlugin
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from utils import image_sequence
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.measure import regionprops
from concurrent.futures import ThreadPoolExecutor
import scipy.ndimage as ndimage


# Define a class that implements the PluginInterface
class Plugin(BasePlugin):
    # Prompt the user to select a file
    def run(self):
        """
        Prompt the user to select a file and process the selected file.

        Raises:
            FileNotFoundError: If no file is selected.

        Example:
            run()
        """

        # Open a file dialog to select a file
        file_path = self.select_folder("Select Folder")

        # Open a file dialog to select a folder to save the centered volume
        save_path = self.select_folder("Select Folder to save centered volume")

        # Check if a file was selected
        if file_path:
            print("the path is: ", file_path)

            # Create a progress window

            self.update_progress(0, "Loading Volume")

            volume = image_sequence.read_sequence2(file_path, progress_window=self)

            self.update_progress(100, "Loading Volume", 1, 1)

            # Select thresholds
            user_inputs = self.request_gui(self.get_thresholds, volume)

            # Perform operations on the selected file based on user inputs
            if len(user_inputs) > 0:
                # main
                # Create a progress window

                self.update_progress(0, "Centering Volume")
                # get shift
                shift = user_inputs["Main"]
                # shift volume
                volume = self.shift_volume_concurrent(
                    volume, shift, progress_window=self
                )

                # select dimensions for cropping
                user_inputs2 = self.request_gui(self.get_dimensions, volume)

                if user_inputs2:
                    # cropping

                    volume = self.crop_volume(
                        volume, user_inputs2["Main"][0], user_inputs2["Main"][1]
                    )

                    # Check if a folder was selected
                    if save_path:
                        # Create a progress window

                        # Save the centered volume
                        self.update_progress(0, "Saving centered Volume")
                        image_sequence.write_sequence2(
                            save_path,
                            os.path.basename(save_path),
                            volume,
                            progress_window=self,
                        )

                        # Show a message box if the centered volume is saved successfully, when the message box is closed, close all matplotlib figures
                        self.prompt_message("centered volume saved successfully.")
                        self.request_gui(plt.close, "all")

                        self.update_progress(100, "Saving centered Volume", 1, 1)

            else:
                # Show a message box if user inputs are not provided
                self.prompt_error("No input detected.")

        else:
            self.prompt_error("No file selected.")

        return 0

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

        # detect if the volume is 8bits or 16bits
        if volume.dtype == np.uint8:
            is8bit = True
        else:
            is8bit = False

        names = ["Main"]

        user_inputs = {"Main": None}

        if is8bit:
            top_threshold = 255
        else:
            top_threshold = 65535

        for name in names:
            while True:
                top_threshold = self.get_user_inputs(name, default_value=top_threshold)

                # Check if the threshold is valid by aplying it to the middle slice
                if top_threshold != None:
                    middle_slice = volume[len(volume) // 2]

                    top_index = np.where(middle_slice > top_threshold)

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

                    thresholded_slice[top_index] = 0

                    # Label the objects in the thresholded slice
                    labeled_slice = label(thresholded_slice)

                    # Get the properties of each labeled region
                    regions = regionprops(labeled_slice)

                    if regions == []:
                        # Show a message box if invalid threshold is selected
                        msg_box = QMessageBox(self.main_window)
                        msg_box.setText("To low threshold selected.")
                        msg_box.exec()
                        continue

                    # Find the largest connected component
                    largest_component = max(regions, key=lambda region: region.area)

                    # Create a mask to keep only the largest component
                    mask = np.zeros_like(labeled_slice)
                    mask[labeled_slice == largest_component.label] = 1

                    # find the center of mass of the largest component
                    center_of_mass = ndimage.measurements.center_of_mass(mask)

                    # center the image in the center of mass
                    center = np.array(middle_slice.shape) // 2
                    shift = np.array(center) - np.array(center_of_mass)
                    shifted_slice = ndimage.shift(mask, shift)

                    # Print the center of mass
                    print(f"The center of the largest component is {center_of_mass}.")

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
                    plt.imshow(shifted_slice, cmap="gray")
                    plt.title("Centered Slice")

                    # Adjusting the layout and displaying the plot
                    plt.tight_layout()
                    plt.show()

                    plt.show()

                    # Ask the user if the threshold is ok
                    msg_box = QMessageBox(self.main_window)
                    msg_box.setText(f"Is the threshold ok for {name} reslice?")
                    msg_box.setStandardButtons(
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)
                    ret = msg_box.exec()

                    # close the figure
                    plt.close()

                    if ret == QMessageBox.StandardButton.Yes:
                        user_inputs[name] = shift
                        break
                    else:
                        continue
                else:
                    # Show a message box if no threshold is selected
                    msg_box = QMessageBox(self.main_window)
                    msg_box.setText("No threshold selected.")
                    msg_box.exec()

        return user_inputs

    def get_dimensions(self, volume):
        """
        Get the dimensions for different slices of a volume.

        Args:
        - self: the object instance
        - volume: a 3D numpy array representing the volume

        Returns:
        - user_inputs: a dictionary containing the dimensions for different slices

        Raises:
        - No specific exceptions are raised within this function.

        Example:
        ```python
        # Create an instance of the class
        instance = ClassName()

        # Define a 3D numpy array representing the volume
        volume = np.random.rand(10, 10, 10)

        # Call the function to get the dimensions
        dimensions = instance.get_dimensions(volume)
        ```
        """

        names = ["Main"]

        user_inputs = {"Main": None}

        for name in names:
            while True:
                dimensions = self.get_user_inputs2(
                    name, default_values=[volume.shape[1], volume.shape[2]]
                )

                # Check if the dimensions are valid by aplying it to the middle slice
                if dimensions != None:
                    z, old_x, old_y = volume.shape

                    x, y = dimensions

                    # Calculate the starting and ending indices for cropping

                    start_x = (old_x - x) // 2
                    start_y = (old_y - y) // 2
                    end_x = start_x + x
                    end_y = start_y + y

                    # Try it in the middle slice

                    middle_slice = volume[len(volume) // 2]

                    # Crop the middle slice
                    cropped_slice = middle_slice[start_x:end_x, start_y:end_y]

                    # plot in the same plot middle_slice and cropped_slice usign matplotlib
                    plt.figure()
                    # Plotting the original middle slice
                    plt.subplot(2, 2, 1)
                    plt.imshow(middle_slice, cmap="gray")
                    plt.title("Original Middle Slice")

                    # Plotting the thresholded slice
                    plt.subplot(2, 2, 2)
                    plt.imshow(cropped_slice, cmap="gray")
                    plt.title("Thresholded Slice")

                    plt.show()

                    # Ask the user if the threshold is ok
                    msg_box = QMessageBox(self.main_window)
                    msg_box.setText(f"Are the dimensions ok for {name} reslice?")
                    msg_box.setStandardButtons(
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)
                    ret = msg_box.exec()

                    if ret == QMessageBox.StandardButton.Yes:
                        user_inputs[name] = dimensions
                        plt.close()
                        break
                    else:
                        plt.close()
                        continue
                else:
                    # Show a message box if no threshold is selected
                    msg_box = QMessageBox(self.main_window)
                    msg_box.setText("No dimensions selected.")
                    msg_box.exec()

        return user_inputs

    def get_user_inputs(self, name, default_value=65535):
        # Get the threshold value for the name reslice
        threshold, ok = QInputDialog.getInt(
            self.main_window,
            "Threshold",
            f"Enter Threshold for {name} reslice:",
            default_value,
        )

        if ok:
            return threshold
        else:
            return None

    def get_user_inputs2(self, name, default_values=[0, 0]):
        # Get the threshold value for the name reslice
        x, ok = QInputDialog.getInt(
            self.main_window, "X", f"Enter X for {name} reslice:", default_values[0]
        )
        y, ok = QInputDialog.getInt(
            self.main_window, "Y", f"Enter Y for {name} reslice:", default_values[1]
        )

        if ok:
            return x, y
        else:
            return None

    def crop_volume(self, volume, x, y):
        """
        Crop a volume to fit (x, y) dimensions.

        Args:
        volume (numpy.ndarray): A 3D array representing the volume.
        x (int): The desired x dimension.
        y (int): The desired y dimension.

        Returns:
        numpy.ndarray: The cropped volume.
        """
        z, old_x, old_y = volume.shape

        # Calculate the starting and ending indices for cropping

        start_x = (old_x - x) // 2
        start_y = (old_y - y) // 2
        end_x = start_x + x
        end_y = start_y + y

        # Crop the volume
        cropped_volume = volume[:, start_x:end_x, start_y:end_y]

        return cropped_volume

    def shift_volume_concurrent(self, volume, shift, progress_window=None):
        """
        Apply autothreshold_slice to each slice in a volume concurrently with a progress bar.

        Args:
        volume (numpy.ndarray): A 3D array where each slice corresponds to an image.

        Returns:
        numpy.ndarray: A 3D array where each slice is shifted.
        """

        def shift_slice(args):
            """
            Shifts a slice by a given shift value.

            Args:
            slice_data (numpy.ndarray): The input slice.
            shift (tuple): The shift value in each dimension.

            Returns:
            numpy.ndarray: The shifted slice.
            """

            slice_data, shift = args

            shifted_slice = ndimage.shift(slice_data, shift, mode="constant", cval=0)
            return shifted_slice

        shifted_volume = np.zeros_like(volume)

        if progress_window == None:
            with ThreadPoolExecutor() as executor:
                args = [(volume[i], shift) for i in range(volume.shape[0])]
                for i, result in enumerate(executor.map(shift_slice, args)):
                    shifted_volume[i] = result
        else:
            with ThreadPoolExecutor() as executor:
                args = [(volume[i], shift) for i in range(volume.shape[0])]
                for i, result in enumerate(executor.map(shift_slice, args)):
                    shifted_volume[i] = result
                    progress_window.update_progress(
                        int(i / volume.shape[0] * 100),
                        f"Shifting: {i}",
                        i,
                        volume.shape[0],
                    )

        return np.array(shifted_volume)
