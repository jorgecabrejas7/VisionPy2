import numpy as np

from base_plugin import BasePlugin
from plugins.Equalizer_4__4.tools import (
    create_equalization_settings_dialog,
    create_file_settings_dialog,
    create_slice_range_dialog,
    calculate_min,
    calculate_max,
)
import traceback
from utils.image_sequence import read_virtual_sequence
from utils.bit_depth import convert_to_8bit
from utils.contrast_and_brightness import auto_adjust, equalize
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from PyQt6.QtWidgets import QMessageBox, QDialog, QLineEdit
import logging
import os

# create a logger that writes to a file called eq.log using python logging
# Create or get the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the debug level or higher as needed

log_file_path = os.path.join("./", "eq.log")
print(f"Logging to: {log_file_path}")  # Debug print to check path

try:
    # Create a file handler which logs even debug messages
    fh = logging.FileHandler(log_file_path, mode="w")
    fh.setLevel(logging.DEBUG)

    # Create formatter and add it to the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(threadName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(fh)

    logger.debug("This is a debug message.")  # First message to trigger file creation

except PermissionError:
    print("Error: Permission denied to write to the log file.")
except Exception as e:
    print(f"An error occurred: {e}")


DEBUG = True


class Plugin(BasePlugin):
    def execute(self):
        try:
            # Prompt the user to select a folder to load the volume from
            folder_path = self.select_folder("Select folder to load volume from")
            if not folder_path:
                self.prompt_error("No folder selected")
                self.finished.emit(self)
                return

            # Read the volume from the selected folder
            volume = read_virtual_sequence(folder_path)

            # Prompt the user to select equalization settings
            self.result = self.request_gui(
                create_equalization_settings_dialog(self.main_window)
            )
            if not self.result:
                self.prompt_error("No parameters selected")
                self.finished.emit(self)
                return

            logger.debug(f"{self.result = }")

            # Prompt the user to select file settings
            file_settings = self.request_gui(
                create_file_settings_dialog(self.main_window)
            )
            if not file_settings:
                self.prompt_error("No file settings selected")
                self.finished.emit(self)
                return

            logger.debug(f"{file_settings = }")

            # Handle file settings
            if file_settings == "duplicate":
                mode = "r"
                # Prompt the user to select a folder to save the volume to
                save_path = self.select_folder("Select folder to save volume to")
            if file_settings == "apply":
                mode = "r+"

            # Check if the start and end slices are within the volume bounds
            if (
                self.result["start_slice"] < 1
                or self.result["end_slice"] > volume.shape[0]
            ):
                self.prompt_error("Slice out of bounds")
                self.finished.emit(self)
                return

            # Handle ROI selection for material and background
            mat_roi = self.handle_roi_selection(volume, "mat")
            bkg_roi = self.handle_roi_selection(volume, "bkg")

            # Update the progress bar
            self.update_progress(0, "Processing...", 0, volume.shape[0])

            # Process each slice in the volume
            self.process_volume(volume, mat_roi, bkg_roi, save_path, file_settings)
        except Exception as e:
            logger.exception(f"Error during execution: {e}")

    def process_volume(self, volume, mat_roi, bkg_roi, save_path, file_settings):
        # Iterate over each slice in the volume
        try:
            for slice_index, _slice in enumerate(volume):
                # Process the slice and obtain the 8-bit version
                try:
                    slice_8bit = self.process_slice(
                        _slice,
                        mat_roi,
                        bkg_roi,
                        slice_index,
                        self.result["threshold_mat"],
                        self.result["threshold_bkg"],
                        volume,
                    )

                    # Check the file settings
                    if file_settings == "duplicate":
                        # Save the 8-bit slice as a TIFF file in the specified folder
                        tifffile.imwrite(
                            f"{save_path}/eq_slice_{slice_index}.tif", slice_8bit
                        )
                    elif file_settings == "apply":
                        # Update the original volume with the 8-bit slice
                        volume[slice_index] = slice_8bit
                except Exception as e:
                    logger.exception(f"Error processing slice {slice_index}: {e}")
        except Exception as e:
            logger.exception(f"Error processing slice {slice_index}: {e}")
        finally:
            self.update_progress(
                100, "Processing complete", volume.shape[0], volume.shape[0]
            )

    def process_slice(
        self,
        _slice,
        mat_roi,
        bkg_roi,
        slice_index,
        threshold_mat,
        threshold_bkg,
        volume,
    ):
        
        # Determine the material and background ROI for the current slice
        # If the user has not selected to manually select the ROI, use the predefined ROI
        # Otherwise, create a copy of the mask ROI
        slice_mat = (
            bool(mat_roi[slice_index])
            if not self.result["select_roi_mat"]
            else mat_roi.copy()
        )
        slice_bkg = (
            bool(bkg_roi[slice_index])
            if not self.result["select_roi_bkg"]
            else bkg_roi.copy()
        )

        # Calculate the peak values for the material and background
        mat_val = self.calculate_peak(_slice, slice_mat, threshold_mat)
        bkg_val = self.calculate_peak(_slice, slice_bkg, threshold_bkg)

        error_mat, error_bkg, average_error, n_it = 2, 2, 2, 0

        # Calculate initial min and max values
        initial_min = calculate_min(
            mat_val,
            bkg_val,
            self.result["target_material"],
            self.result["target_background"],
        )
        initial_max = calculate_max(
            mat_val,
            bkg_val,
            self.result["target_material"],
            self.result["target_background"],
        )

        # Create axuiliary and last generated min and max values
        aux_min, aux_max = initial_min, initial_max

        # Initialize best min, max and error values
        best_min, best_max, best_error = initial_min, initial_max, average_error

        # Iterate until convergence or maximum iterations
        while (
            average_error > self.result["error_threshold"]
            and n_it < self.result["max_it"]
        ):
            logger.debug(
                f"Slice {slice_index} - Iteration {n_it} Mat: {mat_val}, Bkg: {bkg_val}"
            )

            # Set min and max values for slices ot of the equalization range
            if slice_index == self.result["start_slice"] - 1:
                self.first_slice_min = aux_min
                self.first_slice_max = aux_max

            if slice_index == self.result["end_slice"] - 1:
                self.last_slice_min = aux_min
                self.last_slice_max = aux_max

            # Equalize the slice with current calculated min and max values
            equalized_slice = equalize(_slice, aux_min, aux_max)

            # Calculate peak values for material and bakground for the equalized slice
            mat_val_8bit = self.calculate_peak(
                equalized_slice, slice_mat, threshold_mat
            )
            bkg_val_8bit = self.calculate_peak(
                equalized_slice, slice_bkg, threshold_bkg
            )

            # Calculate errors
            error_mat = self.result["target_material"] - mat_val_8bit
            error_bkg = self.result["target_background"] - bkg_val_8bit

            average_error = (abs(error_mat) + abs(error_bkg)) / 2

            logger.debug(
                f"Slice {slice_index} Error Mat: {error_mat}, Error Bkg: {error_bkg}, Average Error: {average_error} Min: {aux_min}, Max: {aux_max}"
            )

            # Update best metrics before updating
            if best_error > average_error:
                best_min, best_max, best_error = aux_min, aux_max, average_error

            # If convergence is not achieved, update min and max values
            if average_error > self.result["error_threshold"]:
                aux_min, aux_max = (
                    round(aux_min - 0.25 * error_mat * 65535 / 255),
                    round(aux_max - 0.25 * error_bkg * 65535 / 255),
                )

            n_it += 1

        debugging_string = (
            f"Slice {slice_index} - Final Min: {aux_min}, Final Max: {aux_max} Average Error: {average_error}"
            if average_error <= self.result["error_threshold"]
            else f"WARNING: Slice {slice_index} - Convergence not achieved after {n_it} iterations. Average Error: {average_error} Min: {aux_min}, Max: {aux_max}"
        )

        logger.debug(debugging_string)

        if average_error > best_error:
            aux_min, aux_max = best_min, best_max
            logger.debug(
                f"Slice {slice_index}. Current error is higher than best error. Using best values: Min: {aux_min}, Max: {aux_max} Best Error: {best_error} Current Error: {average_error}"
            )

        if (
            self.result["start_slice"] - 1
            <= slice_index
            <= self.result["end_slice"] - 1
        ):
            min_val, max_val = aux_min, aux_max

        else:
            min_val, max_val = (
                self.first_slice_min,
                self.first_slice_min
                if slice_index < self.result["start_slice"] - 1
                else self.last_slice_min,
                self.last_slice_max,
            )

        self.update_progress(
            int((slice_index / volume.shape[0]) * 100),
            f"Processing slice {slice_index} of {volume.shape[0]}",
            slice_index,
            volume.shape[0],
        )
        return equalize(_slice, min_val, max_val)

    def handle_roi_selection(self, volume, roi_type):
        """
        Handles the ROI selection by determining if the ROI should be manually defined or loaded from a predefined mask,
        based on user configuration.

        Parameters:
            volume (ndarray): The 3D volume data which the ROI will be applied to.
            roi_type (str): Specifies the type of ROI, either 'mat' for material or 'bkg' for background.

        Returns:
            ndarray: A boolean mask representing the ROI, or None if an error occurs.
        """
        # Check if the ROI should be manually selected or loaded from a predefined mask
        fixed_roi = self.result[f"select_roi_{roi_type}"]

        if fixed_roi:
            # Manually select the ROI
            roi_mask = self.select_roi_manually(volume, roi_type)
            if roi_mask is None:
                self.prompt_error(f"No {roi_type.upper()} ROI selected manually")
                return None
            return roi_mask
        else:
            # Load ROI from a predefined mask file
            roi_path = self.select_file(f"Select {roi_type.upper()} Mask")
            if not roi_path:
                self.prompt_error(f"No {roi_type.upper()} Mask selected")
                return None
            return self.read_virtual_sequence(roi_path)

    def select_roi_manually(self, volume, roi_type):
        """
        Function to be implemented that allows user to manually select an ROI.

        Parameters:
            volume (ndarray): The volume from which a slice is displayed for ROI selection.
            roi_type (str): 'mat' for material or 'bkg' for background, used for guiding the user.

        Returns:
            ndarray: A boolean mask representing the manually selected ROI.
        """

        # Compute the maximum intensity projection along the z-axis
        res = self.request_gui(create_slice_range_dialog(self.main_window))
        if roi_type == "mat":
            projection = np.min(volume[res["start_slice"] : res["end_slice"]], axis=0)

        elif roi_type == "bkg":
            projection = np.max(volume[res["start_slice"] : res["end_slice"]], axis=0)

        projection = auto_adjust(convert_to_8bit(projection))
        roi = self.get_image_bbox(projection)

        x1, y1, x2, y2 = roi
        roi = np.zeros_like(volume[0])
        roi[y1:y2, x1:x2] = 1
        roi = roi.astype(bool)
        return roi

    def calculate_peak(self, _slice, roi, threshold):
        # Calculate the histogram of the slice
        hist, bins = np.histogram(
            _slice[roi],
            bins=256,
        )

        # Find the maximum value in the histogram
        max_val = np.max(hist)

        # Calculate the threshold value based on the maximum value and the user-defined threshold
        thres = max_val * threshold

        # Calculate the bin centers
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Create a mask based on the histogram values above the threshold
        mask = hist > thres

        # Calculate the weighted sum of the histogram values above the threshold
        weighted_sum = np.sum(hist[mask] * bin_centers[mask])

        # Calculate the total counts of the histogram values above the threshold
        total_counts = np.sum(hist[mask])

        # Calculate the average value by dividing the weighted sum by the total counts
        return weighted_sum / total_counts if total_counts > 0 else 0
