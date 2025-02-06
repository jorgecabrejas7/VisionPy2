from uuid import UUID
import numpy as np
import threading
import queue

from base_plugin import BasePlugin
from plugins.Equalizer_4__4.tools import (
    create_equalization_settings_dialog,
    create_file_settings_dialog,
    create_slice_range_dialog,
    calculate_min,
    calculate_max,
)
from utils.image_sequence import read_virtual_sequence
from utils.bit_depth import convert_to_8bit
from utils.contrast_and_brightness import equalize
import tifffile
import logging
import os


from views.main_window import MainWindow

# create a logger that writes to a file called eq.log using python logging
# Create or get the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the debug level or higher as needed

log_file_path = os.path.join("./", "eq.log")
print(f"Logging to: {log_file_path}")  # Debug print to check path

try:
    # Create a file handler which logs even debug messages
    fh = logging.FileHandler(log_file_path, mode="a")
    fh.setLevel(logging.DEBUG)

    # Create formatter and add it to the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(threadName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)

    # Add the handler to the #logger
    logger.addHandler(fh)

    logger.debug("This is a debug message.")  # First message to trigger file creation

except PermissionError:
    print("Error: Permission denied to write to the log file.")
except Exception as e:
    print(f"An error occurred: {e}")


DEBUG = True


class Plugin(BasePlugin):
    def __init__(self, main_window: MainWindow, plugin_name: str, uuid: UUID):
        super().__init__(main_window, plugin_name, uuid)
        self.last_min, self.last_max, self.last_slice = None, None, None
        self.counter = [0]

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
            self.total_slices = volume.shape[0]
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

    # Functions for the processing logic
    # These functions are called by the execute function
    # They contain the logic for processing the volume and slices
    # They should be implemented by the plugin developer
    def process_volume(self, volume, mat_roi, bkg_roi, save_path, file_settings):
        total_slices = volume.shape[0]
        processing_queue = queue.Queue()
        saving_queue = queue.Queue()
        stop_event = threading.Event()
        lock = threading.Lock()

        def processing_worker():
            while not stop_event.is_set():
                try:
                    index, _slice = processing_queue.get(timeout=1)
                    slice_8bit = self.process_slice(
                        _slice,
                        mat_roi,
                        bkg_roi,
                        index,
                        self.result["threshold_mat"],
                        self.result["threshold_bkg"],
                        volume,
                    )
                    saving_queue.put((index, slice_8bit))
                    processing_queue.task_done()
                except queue.Empty:
                    continue

        def saving_worker():
            while not stop_event.is_set():
                try:
                    index, slice_8bit = saving_queue.get(timeout=1)
                    if file_settings == "duplicate":
                        tifffile.imwrite(
                            f"{save_path}/eq_slice_{index}.tif", slice_8bit
                        )
                    elif file_settings == "apply":
                        volume[index] = slice_8bit
                    saving_queue.task_done()
                    with lock:
                        self.counter[0] += 1
                    self.update_progress(
                        int((self.counter[0] + 1) * 100 / total_slices),
                        f"Processing slice {self.counter[0] + 1} of {self.total_slices}",
                        index + 1,
                        total_slices,
                    )
                except queue.Empty:
                    continue

        num_processing = 6
        num_saving = 2
        for _ in range(num_processing):
            threading.Thread(target=processing_worker, daemon=True).start()
        for _ in range(num_saving):
            threading.Thread(target=saving_worker, daemon=True).start()

        for index, _slice in enumerate(volume):
            processing_queue.put((index, _slice))

        processing_queue.join()
        saving_queue.join()
        stop_event.set()
        self.update_progress(100, "Processing complete", total_slices, total_slices)

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
        """
        TODO:
        - Implement a boolean to choose to do histogram matching with last slice if convergence has not been achieved.
        - Implement testing with one or few slices to select optimal params.
        - In testing, let the user try until they are satisfied with the result, asking whether to plot the results.
        """
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
            # Set min and max values for slices ot of the equalization range
            if slice_index == self.result["start_slice"] - 1:
                self.first_slice_min = aux_min
                self.first_slice_max = aux_max

            if slice_index == self.result["end_slice"] - 1:
                self.last_slice_min = aux_min
                self.last_slice_max = aux_max

            # Calculate the error values for the equalized slice
            average_error, error_mat, error_bkg = self.calculate_error(
                _slice, slice_mat, slice_bkg, aux_min, aux_max
            )

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

        if self.last_min is not None and self.last_max is not None:
            if (
                best_error
                > self.calculate_error(
                    _slice, slice_mat, slice_bkg, self.last_min, self.last_max
                )[0]
            ):
                # Using last known good values
                aux_min, aux_max = self.last_min, self.last_max
            else:
                # Using best values from current computations
                aux_min, aux_max = best_min, best_max

        else:
            # No last values available, use the best current values
            aux_min, aux_max = best_min, best_max

        # Set the new last values
        new_error = self.calculate_error(
            _slice, slice_mat, slice_bkg, aux_min, aux_max
        )[0]
        self.last_min, self.last_max = aux_min, aux_max

        if (
            self.result["start_slice"] - 1
            <= slice_index
            <= self.result["end_slice"] - 1
        ):
            min_val, max_val = aux_min, aux_max

        else:
            if slice_index < self.result["start_slice"] - 1:
                min_val = self.first_slice_min
                max_val = self.first_slice_min
            else:
                min_val = self.last_slice_min
                max_val = self.last_slice_max

        self.update_progress(
            int((slice_index / volume.shape[0]) * 100),
            f"Processing slice {slice_index} of {volume.shape[0]}",
            slice_index,
            volume.shape[0],
        )

        if (
            self.result["histogram_matching"]
            and new_error > self.result["error_threshold"]
        ):
            slice_8bit = self.histogram_match(_slice, self.last_slice)
        else:
            slice_8bit = equalize(_slice, min_val, max_val)

        self.last_slice = slice_8bit

        return slice_8bit

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
        # Request a slice range from the user via the GUI.
        res = self.request_gui(create_slice_range_dialog(self.main_window))
        start_slice, end_slice = res["start_slice"], res["end_slice"]
        total_slices = end_slice - start_slice  # Total slices to process

        import concurrent.futures
        import threading
        import numpy as np

        # Shared progress counter using a mutable container.
        progress_counter = [
            0
        ]  # We'll store the count as the first element of this list.
        progress_lock = threading.Lock()

        num_workers = 4
        # Split the slice indices among the available threads.
        indices = np.array_split(range(start_slice, end_slice), num_workers)

        def compute_projection(index_range):
            if not len(index_range):
                return None

            # Process the first slice in the chunk.
            proj = volume[index_range[0]]

            # Process the remaining slices in this chunk.
            for i in index_range[1:]:
                slice_data = volume[i]
                if roi_type == "mat":
                    proj = np.minimum(proj, slice_data)
                elif roi_type == "bkg":
                    proj = np.maximum(proj, slice_data)

                # Update progress after processing each slice.
                with progress_lock:
                    progress_counter[0] += 1
                    current_count = progress_counter[0]
                self.update_progress(
                    (current_count + 1) * 100 / total_slices,
                    f"Processing slice {current_count}",
                    current_count,
                    total_slices,
                )

            return proj

        # Launch parallel threads to compute parts of the projection.
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(compute_projection, chunk) for chunk in indices]
            # Collect results as they complete.
            partial_projs = [
                future.result()
                for future in concurrent.futures.as_completed(futures)
                if future.result() is not None
            ]

        # Combine the partial projections into a final projection.
        if partial_projs:
            projection = partial_projs[0]
            for part in partial_projs[1:]:
                if roi_type == "mat":
                    projection = np.minimum(projection, part)
                elif roi_type == "bkg":
                    projection = np.maximum(projection, part)
        else:
            projection = volume[start_slice]

        self.update_progress(100, "Z-Projection Complete", total_slices, total_slices)

        # Use the computed projection to determine the ROI bounding box.
        roi_bbox = self.get_image_bbox(projection)
        x1, y1, x2, y2 = roi_bbox

        # Create a boolean mask of the ROI.
        roi_mask = np.zeros_like(volume[0])
        roi_mask[y1:y2, x1:x2] = 1
        return roi_mask.astype(bool)

    def calculate_peak(self, _slice, roi, threshold):
        # Calculate the histogram of the slice
        hist, bins = np.histogram(
            _slice[roi],
            bins=256,
        )

        # Calculate the threshold value based on the maximum value and the user-defined threshold
        thres = np.max(hist) * threshold

        # Calculate the bin centers
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Create a mask based on the histogram values above the threshold
        mask = hist > thres

        # Calculate weighted sum and total counts in a single step using masked arrays
        masked_hist = hist[mask]
        masked_centers = bin_centers[mask]
        weighted_sum = np.dot(masked_hist, masked_centers)
        total_counts = masked_hist.sum()

        # Calculate the average value by dividing the weighted sum by the total counts
        return weighted_sum / total_counts if total_counts > 0 else 0

    def calculate_error(self, slice_data, roi_mat, roi_bkg, min_val, max_val):
        # Calculate average error for a slice  with certain min and max values
        data = equalize(slice_data, min_val, max_val)
        mat_val = self.calculate_peak(data, roi_mat, self.result["threshold_mat"])
        bkg_val = self.calculate_peak(data, roi_bkg, self.result["threshold_bkg"])

        error_mat = self.result["target_material"] - mat_val
        error_bkg = self.result["target_background"] - bkg_val

        average_error = (abs(error_mat) + abs(error_bkg)) / 2

        return average_error, error_mat, error_bkg

    def histogram_match(self, target, template):
        """
        Adjust the pixel values of a grayscale image such that its histogram
        matches that of a target image

        Arguments:
        -----------
            source: np.ndarray
                Image to transform; the histogram is computed over the flattened
                array
            template: np.ndarray
                Template image; can have different dimensions to source
        Returns:
        -----------
            matched: np.ndarray
                The transformed output image
        """

        source = convert_to_8bit(target, use_scaling=True)
        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()

        # get the set of unique pixel values and their corresponding indices and counts
        s_values, bin_idx, s_counts = np.unique(
            source, return_inverse=True, return_counts=True
        )
        t_values, t_counts = np.unique(template, return_counts=True)

        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

        # Use the interpolated values as indices to reassign pixels in the source
        matched = interp_t_values[bin_idx].reshape(oldshape).astype(source.dtype)

        return matched
