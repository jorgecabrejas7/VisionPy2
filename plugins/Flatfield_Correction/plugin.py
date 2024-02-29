# plugins/Flatfield%20Correction/plugin.py

import os
import traceback
from typing import *

import cv2
import numpy as np
import tifffile
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from pystackreg.util import to_uint16

from base_plugin import BasePlugin
from utils.bit_depth import f32_to_uint16
from utils.image_sequence import read_virtual_sequence
from utils.image_utils import read_tif
from utils.register import stackreg_translate


class Plugin(BasePlugin):
    def __init__(self, main_window, plugin_name):
        super().__init__(main_window, plugin_name)

    def execute(self):
        try:
            folder = self.select_folder(caption="Select Folder to load volume")
            if not folder:
                self.prompt_error("No folder selected")
                self.finished.emit(self)
                return

            original_slice_names = sorted(
                [file for file in os.listdir(folder) if file.endswith(".tif")]
            )

            volume = read_virtual_sequence(folder)
            slice_n, bbox = self.get_volume_bbox(volume)
            print(f"{folder = } {slice_n = } {bbox = }")
            flatfield_path = self.select_file(caption="Select Flatfield image")
            ddm_path = self.select_file(caption="Select DDM image")
            flatfield = read_tif(flatfield_path)
            ddm = read_tif(ddm_path)
            flatfield_copy = flatfield.copy()

            erode_kernel = np.ones((3, 3), np.uint8)
            erode_1 = cv2.erode(ddm, erode_kernel, iterations=1)
            erode_2 = cv2.erode(erode_1, erode_kernel, iterations=2)
            contour_mask = cv2.bitwise_not(erode_1) - cv2.bitwise_not(ddm)

            contours, _ = cv2.findContours(
                contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            for contour in contours:
                # Specify the desired thickness for the contour
                contour_thickness = 3  # You can adjust this value as needed

                # Create a mask with the thicker contour
                contour_mask = np.zeros_like(ddm)
                cv2.drawContours(
                    contour_mask, [contour], -1, color=255, thickness=contour_thickness
                )

                # Calculate the average gray level value on the thicker contour
                mean_val = cv2.mean(flatfield_copy, mask=contour_mask)

                # Apply the mean value to the area inside the original contour
                # Use a filled contour here to fill the entire area
                cv2.fillPoly(flatfield_copy, pts=[contour], color=mean_val)

            contours, _ = cv2.findContours(
                erode_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                # Create a mask for the current contour
                contour_mask = np.zeros_like(flatfield_copy)
                cv2.drawContours(
                    contour_mask, [contour], -1, color=255, thickness=-1
                )  # Filled contour

                # Apply Gaussian blur to the entire image (for simplicity)
                blurred_image = cv2.GaussianBlur(flatfield_copy, (5, 5), 0)

                # Replace only the contour area in the original image with the blurred one
                flatfield_copy = np.where(
                    contour_mask == 255, blurred_image, flatfield_copy
                )

            x1, y1, x2, y2 = bbox
            flat_roi = flatfield_copy[y1:y2, x1:x2].copy()
            vol_roi = np.array(volume[slice_n, y1:y2, x1:x2]).copy()

            sr = stackreg_translate(vol_roi, flat_roi)
            # Apply the translation to the flatfield
            flatfield_copy = sr.transform(flatfield_copy)
            flatfield_copy = to_uint16(flatfield_copy)

            # Copy the detector defects back to the corrected and translated flatfield
            inverted_ddm = (255 - ddm).astype(bool)  # Convert to boolean mask
            flatfield_copy[inverted_ddm] = flatfield[inverted_ddm]

            # Get folder to save the data
            save_folder = self.select_folder(caption="Select folder to save results")

            # Add correction for loading partial volume (centered), transpose, show, check line, calculate angle and get everything inside the for loop  into a function and paralellize with dask

            if not save_folder:
                self.prompt_error("No folder selected")
                self.finished.emit(self)
                return

            num_slices = len(original_slice_names)
            for index, name in enumerate(original_slice_names):
                self.update_progress(
                    int(index / num_slices * 100),
                    f"Processing slice {index + 1} of {num_slices}",
                    index,
                    num_slices,
                )

                _slice = volume[index]

                # Perform element-wise division of _slice by flatfield_copy
                result = np.divide(
                    _slice.astype(np.float32),
                    flatfield_copy.astype(np.float32),
                    out=_slice.astype(np.float32),
                    where=flatfield_copy.astype(np.float32) != 0.0,
                )
                # Convert the result to uint16 format with scaling
                result = f32_to_uint16(result, do_scaling=True)

                # Write result to disk as a TIFF file
                tifffile.imwrite(os.path.join(save_folder, name), result)

        except Exception as e:
            traceback.print_exc()
            self.prompt_error(f"Error while running Flatfield Correction: {str(e)}")
            self.finished.emit(self)
            return
