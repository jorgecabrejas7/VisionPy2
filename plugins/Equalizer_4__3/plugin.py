import numpy as np

from base_plugin import BasePlugin
from plugins.Equalizer_4__3.tools import (
    create_equalization_settings_dialog,
    create_file_settings_dialog,
    equalX,
    equalY,
)
import traceback
from utils.image_sequence import read_virtual_sequence
from utils.contrast_and_brightness import adjust_brightness_contrast
import tifffile
from read_roi import read_roi_file
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import time
from PyQt6.QtWidgets import QMessageBox

DEBUG = False
class Plugin(BasePlugin):
    def __init__(self, main_window, plugin_name):
        super().__init__(main_window, plugin_name)

    def execute(self):
        # Update the function to integrate with dask for parallel processing
        try:
            start = time.time()
            """
            Show max and min projections of center slices. Max BKg min Mat. oNLY IF FIX BKG AND FIX MAT
            """
            folder_path = self.select_folder("Select folder to load volume from")
            if not folder_path:
                self.prompt_error("No folder selected")
                self.finished.emit(self)
                return
            
            volume = read_virtual_sequence(folder_path)
            result = self.request_gui(create_equalization_settings_dialog(self.main_window))
            if not result:
                    self.prompt_error("No parameters selected")
                    self.finished.emit(self)
                    return
            
            print(f"{result = }")

            file_settings = self.request_gui(create_file_settings_dialog(self.main_window))
            if not file_settings:
                self.prompt_error("No file settings selected")
                self.finished.emit(self)
                return
            
            print(f"{file_settings = }")

            if file_settings == "duplicate":
                mode = "r"
                save_path = self.select_folder("Select folder to save volume to")
            if file_settings == "apply":
                mode = "r+" 

            if result["start_slice"] < 1 or result["end_slice"] > volume.shape[0]:
                self.prompt_error("Slice out of bounds")
                self.finished.emit(self)
                return

            if not result["fix_mat_ROI"]:
                mat_roi_path = self.select_file("Select MAT Mask")
                if not mat_roi_path:
                    self.prompt_error("No MAT Mask selected")
                    self.finished.emit(self)
                    return
                mat_roi = read_virtual_sequence(mat_roi_path)

            else:
                mat_roi_path = self.select_file("Select MAT ROI")
                if not mat_roi_path:
                    self.prompt_error("No MAT ROI selected")
                    self.finished.emit(self)
                    return
                roi = read_roi_file(mat_roi_path)
                aux = list(roi.values())[0]
                print(f"roi_mat: {roi}")
                bbox_mat = aux["left"], aux["top"], aux["left"] + aux["width"], aux["top"] + aux["height"]
                mat_roi = volume[0].copy()
                mat_roi[:, :] = 0
                mat_roi[bbox_mat[1] : bbox_mat[3], bbox_mat[0] : bbox_mat[2]] = 1
                mat_roi = mat_roi.astype(bool)
                

            if not result["fix_bkg_ROI"]:
                bkg_roi_path = self.select_file("Select BKG Mask")
                if not bkg_roi_path:
                    self.prompt_error("No BKG Mask selected")
                    self.finished.emit(self)
                    return
                bkg_roi = read_virtual_sequence(mat_roi_path)

            else:
                bkg_roi_path = self.select_file("Select BKG ROI")
                if not bkg_roi_path:
                    self.prompt_error("No BKG ROI selected") 
                    self.finished.emit(self)
                    return
                roi = read_roi_file(bkg_roi_path)
                print(f"roi_bkg: {roi}")
                aux = list(roi.values())[0]
                bbox_bkg = aux["left"], aux["top"], aux["left"] + aux["width"], aux["top"] + aux["height"]
                bkg_roi = volume[0].copy()
                bkg_roi[:, :] = 0
                bkg_roi[bbox_bkg[1] : bbox_bkg[3], bbox_bkg[0] : bbox_bkg[2]] = 1
                bkg_roi = bkg_roi.astype(bool)

            self.update_progress(0, "Processing...", 0, volume.shape[0])
            for slice_index in range(volume.shape[0]):
                try:    
                    slice_mat = (
                        bool(mat_roi[slice_index])
                        if not result["fix_mat_ROI"]
                        else mat_roi.copy()
                    )
                    slice_bkg = (
                        bool(bkg_roi[slice_index])
                        if not result["fix_bkg_ROI"]
                        else bkg_roi.copy()
                    )

                    current_slice = volume[slice_index]

                    # Material values
                    mat_max = np.max(current_slice[slice_mat])
                    mat_min = np.min(current_slice[slice_mat])

                    # Get material histogram
                    mat_hist, mat_bins = np.histogram(
                        current_slice[slice_mat], bins=256, range=(mat_min, mat_max)
                    )

                    max_mat = np.max(mat_hist)
                    threshold_mat = max_mat * result["t_mat"]
                    bin_centers = (mat_bins[:-1] + mat_bins[1:]) / 2
                    mask = mat_hist > threshold_mat

                    weighted_sum = np.sum(mat_hist[mask] * bin_centers[mask])
                    total_count = np.sum(mat_hist[mask])

                    mat_val = weighted_sum / total_count if total_count > 0 else 0

                    # Repeat process for background
                    bkg_max = np.max(current_slice[slice_bkg])
                    bkg_min = np.min(current_slice[slice_bkg]) 

                    bkg_hist, bkg_bins = np.histogram(
                        current_slice[slice_bkg], bins=256, range=(bkg_min, bkg_max)
                    )

                    max_bkg = np.max(bkg_hist)
                    threshold_bkg = max_bkg * result["t_bkg"]
                    bin_centers = (bkg_bins[:-1] + bkg_bins[1:]) / 2
                    mask = bkg_hist > threshold_bkg

                    weighted_sum = np.sum(bkg_hist[mask] * bin_centers[mask])
                    total_count = np.sum(bkg_hist[mask])

                    bkg_val = weighted_sum / total_count if total_count > 0 else 0

                    # Equalization loop

                    delta_mat = 2  # Difference between ref_mat_original and ref_mat calculated
                    delta_bkg = 2  # Difference between ref_bkg_original and ref_bkg calculated
                    delta_avg = 2  # Difference between delta average and and ref_bkg and ref_mat calculated
                    n_it = 0

                    X = equalX(mat_val, bkg_val, result["ref_mat_original"], result["ref_bkg_original"])
                    Y = equalY(mat_val, bkg_val, result["ref_mat_original"], result["ref_bkg_original"])
                    X1, Y1 = X, Y
                    last_X, last_Y = X, Y   
                    while ( delta_avg > result["delta"]  and  n_it < result["max_it"]):
                        print(f"Slice {slice_index} - Iteration {n_it} - {delta_avg = }")
                        slice_8bit = adjust_brightness_contrast(current_slice, X, Y)
                        if slice_index == result["start_slice"]:
                            X_s = X1
                            Y_s = Y1
                        
                        if slice_index == result["end_slice"]:
                            X_f = X
                            Y_f = Y
            

                        # Repeat process with the histogram calculation for the 8bit image
                        mat_max_8bit = np.max(slice_8bit[slice_mat])
                        mat_min_8bit = np.min(slice_8bit[slice_mat])

                        mat_hist_8bit, mat_bins_8bit = np.histogram(
                            slice_8bit[slice_mat], bins=256, range=(mat_min_8bit, mat_max_8bit)
                        )

                        max_mat_8bit = np.max(mat_hist_8bit)
                        threshold_mat_8bit = max_mat_8bit * result["t_mat"]
                        bin_centers_8bit = (mat_bins_8bit[:-1] + mat_bins_8bit[1:]) / 2
                        mask_mat_8bit = mat_hist_8bit > threshold_mat_8bit


                        weighted_sum_8bit = np.sum(mat_hist_8bit[mask_mat_8bit] * bin_centers_8bit[mask_mat_8bit])
                        total_count_8bit = np.sum(mat_hist_8bit[mask_mat_8bit])

                        mat_val_8bit = weighted_sum_8bit / total_count_8bit if total_count_8bit > 0 else 0

                        if DEBUG:
                            # After calculating the mask
                            

                            # Check the range of bin centers
                            print("Range of Bin Centers for Material (8-bit): Min =", np.min(bin_centers_8bit), ", Max =", np.max(bin_centers_8bit))
                            print("Number of True in Mask:", np.sum(mask_mat_8bit))

                            # Check if the mask is too restrictive
                            if np.sum(mask_mat_8bit) == 0:
                                print("Warning: No values passing the threshold. Check the threshold value.")

                            # After calculating the weighted sum and total count
                            if total_count_8bit == 0:
                                print("Warning: Total count is zero. No values contributing to the histogram.")
                            print("Threshold for Material:", threshold_mat_8bit)
                            print("Max Material Value (8-bit) used for threshold calculation:", max_mat_8bit)

                        # Repeat process for background
                        bkg_max_8bit = np.max(slice_8bit[slice_bkg])
                        bkg_min_8bit = np.min(slice_8bit[slice_bkg])

                        bkg_hist_8bit, bkg_bins_8bit = np.histogram(
                            slice_8bit[slice_bkg], bins=256, range=(bkg_min_8bit, bkg_max_8bit)
                        )

                        max_bkg_8bit = np.max(bkg_hist_8bit)
                        threshold_bkg_8bit = max_bkg_8bit * result["t_bkg"]
                        bin_centers_8bit = (bkg_bins_8bit[:-1] + bkg_bins_8bit[1:]) / 2
                        mask_bkg_8bit = bkg_hist_8bit > threshold_bkg_8bit

                        weighted_sum_8bit = np.sum(bkg_hist_8bit[mask_bkg_8bit] * bin_centers_8bit[mask_bkg_8bit])
                        total_count_8bit = np.sum(bkg_hist_8bit[mask_bkg_8bit])

                        bkg_val_8bit = weighted_sum_8bit / total_count_8bit if total_count_8bit > 0 else 0
                        # After calculating the mask
                        if DEBUG:
                            # Check the range of bin centers
                            print("Range of Bin Centers for Background (8-bit): Min =", np.min(bin_centers_8bit), ", Max =", np.max(bin_centers_8bit))
                            print("Number of True in Mask:", np.sum(mask_bkg_8bit))

                            # Check if the mask is too restrictive
                            if np.sum(mask_bkg_8bit) == 0:
                                print("Warning: No values passing the threshold. Check the threshold value.")

                            # After calculating the weighted sum and total count
                            if total_count_8bit == 0:
                                print("Warning: Total count is zero. No values contributing to the histogram.")
                            print("Threshold for Background:", threshold_bkg_8bit)
                            print("Max Background Value (8-bit) used for threshold calculation:", max_bkg_8bit)

                        if DEBUG:
                            print(f"\tmat_val_8bit = weighted_sum_8bit / total_count_8bit  = {weighted_sum_8bit} / {total_count_8bit} if {total_count_8bit} > 0 else 0 = {mat_val_8bit}")
                            print(f"\tbkg_val_8bit = weighted_sum_8bit / total_count_8bit  = {weighted_sum_8bit} / {total_count_8bit} if {total_count_8bit} > 0 else 0 = {bkg_val_8bit}")


                        delta_mat = result["ref_mat_original"] - mat_val_8bit
                        delta_bkg = result["ref_bkg_original"] - bkg_val_8bit
                        if DEBUG:
                            print(f"\tdelta_mat = result['ref_mat_original'] - mat_val_8bit = {result['ref_mat_original']} - {mat_val_8bit} = {delta_mat}")
                            print(f"\tdelta_bkg = result['ref_bkg_original'] - bkg_val_8bit = {result['ref_bkg_original']} - {bkg_val_8bit} = {delta_bkg}")                        
                        delta_avg = (abs(delta_mat) + abs(delta_bkg)) / 2

                        X1_new = X1 - 0.25 * delta_bkg * 65535 / 255
                        Y1_new = Y1 - 0.25 * delta_mat * 65535 / 255    

                        # Logging the calculations
                        if DEBUG:
                            print(f"\tX1 = round(X1 - 0.25 * delta_bkg * 65535 / 255) = round({X1} - 0.25 * {delta_bkg} * 65535 / 255) = {X1_new}")
                            print(f"\tY1 = round(Y1 - 0.25 * delta_mat * 65535 / 255) = round({Y1} - 0.25 * {delta_mat} * 65535 / 255) = {Y1_new}")

                        # Rounding the values
                        X1 = round(X1_new)
                        Y1 = round(Y1_new)

                        n_it += 1
                    

                    # Apply equalization to slice
                    print(f"Slice {slice_index} - Iteration {n_it} - {delta_avg = }")
                    if delta_avg > result["delta"]:
                        print("Warning: Maximum number of iterations reached. Results may not be accurate. Using last slice X and Y")
                        X1, Y1 = last_X, last_Y
                    if result["start_slice"] <= slice_index <= result["end_slice"]:
                        X, Y = X1, Y1
                        print(f"{X = } {Y = }")
                        slice_8bit = adjust_brightness_contrast(current_slice, X, Y)
                    else:
                        X, Y = (X_s, Y_s) if slice_index < result["start_slice"] else (X_f, Y_f)
                        slice_8bit = adjust_brightness_contrast(current_slice, X, Y)

                    if file_settings == "duplicate":
                        # Logic to save slices 
                        tifffile.imwrite(f"{save_path}/eq_slice_{slice_index}.tif", slice_8bit)

                    if file_settings == "apply":
                        volume[slice_index] = slice_8bit
                    self.update_progress(int((slice_index / volume.shape[0]) * 100), f"Processing slice {slice_index} of {volume.shape[0]}", slice_index, volume.shape[0])
                except Exception as e:
                    traceback.print_exc()
                    print("error, updating")
                    self.update_progress(int((slice_index / volume.shape[0]) * 100), f"Processing slice {slice_index} of {volume.shape[0]}", slice_index, volume.shape[0])
            self.update_progress(100, "Finished", volume.shape[0], volume.shape[0])
            self.request_gui(show_elapsed_time, start, volume)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e), self.get_name())



def show_elapsed_time(start, volume):
    end = time.time()
    elapsed = end - start
    # Create a message box that shows the total elapsed time and the average time by slice
    msg = QMessageBox()
    msg.setWindowTitle("Elapsed Time")
    msg.setText(f"Total elapsed time: {elapsed:.2f} seconds\nAverage time per slice: {elapsed / volume.shape[0]:.2f} seconds")
    msg.exec()
    return None

def plot_histograms(mat_hist, bkg_hist, mat_bins, bkg_bins, title1="Material Histogram", title2="Background Histogram"):
    """
    Plots histograms for the material and background.

    Parameters:
    mat_hist (np.ndarray): Histogram values for material.
    bkg_hist (np.ndarray): Histogram values for background.
    mat_bins (np.ndarray): Bin edges for material histogram.
    bkg_bins (np.ndarray): Bin edges for background histogram.
    title1 (str): Title for the material histogram plot.
    title2 (str): Title for the background histogram plot.
    """
    # Calculate bin centers
    mat_bin_centers = (mat_bins[:-1] + mat_bins[1:]) / 2
    bkg_bin_centers = (bkg_bins[:-1] + bkg_bins[1:]) / 2

    # Create figure and axes
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot material histogram
    ax[0].bar(mat_bin_centers, mat_hist, width=mat_bin_centers[1] - mat_bin_centers[0], color='blue')
    ax[0].set_title(title1)
    ax[0].set_xlabel('Intensity Value')
    ax[0].set_ylabel('Frequency')

    # Plot background histogram
    ax[1].bar(bkg_bin_centers, bkg_hist, width=bkg_bin_centers[1] - bkg_bin_centers[0], color='green')
    ax[1].set_title(title2)
    ax[1].set_xlabel('Intensity Value')
    ax[1].set_ylabel('Frequency')

    # Display the plot
    plt.tight_layout()
    plt.show(block=True)
    return None

def plot_slice_with_bbox(volume_slice, bbox_mat=None, bbox_bkg=None):
    """
    Plots a slice of the volume with bounding boxes for material and background.

    Args:
    volume_slice (np.ndarray): The slice of the volume to be plotted.
    bbox_mat (tuple): The bounding box for material as (xmin, ymin, xmax, ymax). 
                      If None, no bounding box is drawn.
    bbox_bkg (tuple): The bounding box for background as (xmin, ymin, xmax, ymax). 
                      If None, no bounding box is drawn.

    Returns:
    None
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(volume_slice, cmap='gray')

    # Draw material bounding box if provided
    if bbox_mat is not None:
        xmin, ymin, xmax, ymax = bbox_mat
        width = xmax - xmin
        height = ymax - ymin
        rect_mat = plt.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect_mat)

    # Draw background bounding box if provided
    if bbox_bkg is not None:
        xmin, ymin, xmax, ymax = bbox_bkg
        width = xmax - xmin
        height = ymax - ymin
        rect_bkg = plt.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='b', facecolor='none')
        plt.gca().add_patch(rect_bkg)

    plt.axis('off')
    plt.show()


def plot_with_colored_rois_and_bboxes(mat_roi, bkg_roi, mat_bbox, bkg_bbox):
    """
    Plots an image with colored ROIs and bounding boxes for material and background.

    Args:
    mat_roi (np.ndarray): Boolean array representing the material ROI.
    bkg_roi (np.ndarray): Boolean array representing the background ROI.
    mat_bbox (tuple): Bounding box for the material ROI (xmin, ymin, xmax, ymax).
    bkg_bbox (tuple): Bounding box for the background ROI (xmin, ymin, xmax, ymax).
    """
    # Create an RGB image with all black pixels
    rgb_image = np.zeros((*mat_roi.shape, 3), dtype=np.uint8)

    # Set material ROI pixels to yellow (255, 255, 0)
    rgb_image[mat_roi] = [255, 255, 0]

    # Set background ROI pixels to red (255, 0, 0)
    rgb_image[bkg_roi] = [255, 0, 0]

    # For overlapping regions, create a combined color (orange)
    overlap = np.logical_and(mat_roi, bkg_roi)
    rgb_image[overlap] = [255, 165, 0]  # Orange color for overlap

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(rgb_image)
    
    # Add bounding box for material ROI
    mat_rect = patches.Rectangle(
        (mat_bbox[0], mat_bbox[1]), mat_bbox[2] - mat_bbox[0], mat_bbox[3] - mat_bbox[1],
        linewidth=2, edgecolor='blue', facecolor='none')
    ax.add_patch(mat_rect)

    # Add bounding box for background ROI
    bkg_rect = patches.Rectangle(
        (bkg_bbox[0], bkg_bbox[1]), bkg_bbox[2] - bkg_bbox[0], bkg_bbox[3] - bkg_bbox[1],
        linewidth=2, edgecolor='green', facecolor='none')
    ax.add_patch(bkg_rect)

    # Setting additional plot parameters
    ax.axis('off')
    plt.show()