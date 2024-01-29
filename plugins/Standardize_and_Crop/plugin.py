# plugins/Flatfield%20Correction/plugin.py
import logging
import os
import queue
import threading
import traceback

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.exposure import rescale_intensity
from PyQt6.QtCore import *
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import *
from PyQt6.QtWidgets import (
    QDialog,
    QFormLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from base_plugin import BasePlugin
from utils.bit_depth import convert_to_8bit
from utils.image_sequence import read_virtual_sequence


class Plugin(BasePlugin):
    def __init__(self, main_window, plugin_name):
        super().__init__(main_window, plugin_name)
        self.processing_queue = queue.Queue()
        self.saving_queue = queue.Queue()
        self.total_slices = 0
        self.stop_event = threading.Event()

    def execute(self):
        try:
            eq_settings = self.request_gui(
                create_equalization_settings_dialog(self.main_window)
            )
            target_mean = eq_settings["target_mean"]
            target_std = eq_settings["target_std"]
            folder = self.select_folder(caption="Select Folder to load volume")
            if not folder:
                self.prompt_error("No folder selected")
                self.finished.emit(self)
                return

            original_slice_names = sorted(
                [file for file in os.listdir(folder) if file.endswith(".tif")]
            )

            self.volume = read_virtual_sequence(folder)
            self.n_slices = self.volume.shape[0]
            _, bbox = self.get_volume_bbox(self.volume)

            # Get folder to save the data
            self.save_folder = self.select_folder(
                caption="Select folder to save results"
            )

            # Add correction for loading partial volume (centered), transpose, show, check line, calculate angle and get everything inside the for loop  into a function and paralellize with dask

            if not self.save_folder:
                self.prompt_error("No folder selected")
                self.finished.emit(self)
                return

            x1, y1, x2, y2 = bbox
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            for i in range(self.n_slices):
                # Read and crop the slice
                slice = self.volume[i, y1:y2, x1:x2]
                standardized_slice = self.standardize_slice(
                    slice, target_mean=target_mean, target_std=target_std
                )
                slice_8bit = convert_to_8bit(standardized_slice)
                eq_slice = np.empty_like(slice_8bit)
                cv2.intensity_transform.autoscaling(slice_8bit, eq_slice)
                # Save the cropped slice as a page in the TIFF file
                tifffile.imwrite(
                    os.path.join(
                        self.save_folder, "cropped_" + original_slice_names[i]
                    ),
                    eq_slice,
                    imagej=True,
                    compression="ZLIB",
                )

                # Update the custom progress bar
                self.update_progress(
                    int((i / self.n_slices) * 100),
                    f"Processing slice {i + 1}",
                    i,
                    self.n_slices,
                )
            self.update_progress(100, "Finished", self.n_slices, self.n_slices)

        except Exception as e:
            self.prompt_error(traceback.format_exc())
            logging.error("Traceback:", exc_info=True)
            self.finished.emit(self)
            return

    def standardize_slice(self, slice_data, target_mean=32767, target_std=5000):
        # Calculate current mean and standard deviation
        current_mean = np.mean(slice_data)
        current_std = np.std(slice_data)

        # Prevent division by zero in case of a uniform slice
        if current_std == 0:
            return slice_data

        # Standardize slice
        standardized_slice = (
            slice_data - current_mean
        ) / current_std * target_std + target_mean

        # Ensure the values are within 0 to 65535
        standardized_slice = np.clip(standardized_slice, 0, 65535).astype(np.uint16)

        return standardized_slice


def create_equalization_settings_dialog(parent: QMainWindow) -> callable:
    def get_equalization_settings() -> dict[str, float]:
        class CustomDialog(QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowTitle("Equalization Settings")
                self.create_layout()

            def create_layout(self):
                layout = QVBoxLayout(self)
                form_layout = QFormLayout()

                double_validator = QDoubleValidator(
                    notation=QDoubleValidator.Notation.StandardNotation
                )
                double_validator.setLocale(
                    QLocale(QLocale.Language.English, QLocale.Country.UnitedStates)
                )

                self.target_mean_edit = QLineEdit("32767")
                self.target_mean_edit.setValidator(double_validator)
                form_layout.addRow("Target Mean:", self.target_mean_edit)

                self.target_std_edit = QLineEdit("5000")
                self.target_std_edit.setValidator(double_validator)
                form_layout.addRow("Target Standard Deviation:", self.target_std_edit)

                layout.addLayout(form_layout)

                ok_button = QPushButton("OK")
                ok_button.clicked.connect(self.validate_and_accept)
                layout.addWidget(ok_button)

            def validate_and_accept(self):
                if self.validate_inputs():
                    self.accept()

            def validate_inputs(self):
                if not self.validate_field(self.target_mean_edit, "Target Mean", float):
                    return False
                if not self.validate_field(
                    self.target_std_edit, "Target Standard Deviation", float
                ):
                    return False
                return True

            def validate_field(self, field, name, data_type):
                try:
                    data_type(field.text())
                    return True
                except ValueError:
                    QMessageBox.warning(
                        self, "Input Error", f"Invalid input for {name}."
                    )
                    return False

            def get_values(self):
                return {
                    "target_mean": float(self.target_mean_edit.text()),
                    "target_std": float(self.target_std_edit.text()),
                }

        dialog = CustomDialog(parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.get_values()
        else:
            return None

    return get_equalization_settings


def bi_histogram_equalization(img):
    # Calculate the histogram of the whole image
    hist_full, bins = np.histogram(img.flatten(), 65536, [0, 65536])

    # Calculate the median of the intensity values
    median_val = np.median(img)

    # Create masks for splitting the image into two parts
    mask_low = img <= median_val
    mask_high = img > median_val

    # Calculate separate histograms for the low and high intensity parts of the image
    hist_low, _ = np.histogram(img[mask_low], bins=65536, range=(0, 65536))
    hist_high, _ = np.histogram(img[mask_high], bins=65536, range=(0, 65536))

    # Calculate separate CDFs for the low and high histograms
    cdf_low = hist_low.cumsum()
    cdf_low = (cdf_low - cdf_low.min()) * 65535 / (cdf_low.max() - cdf_low.min())

    cdf_high = hist_high.cumsum()
    cdf_high = (cdf_high - cdf_high.min()) * 65535 / (
        cdf_high.max() - cdf_high.min()
    ) + cdf_low.max()

    # Map the pixel values of the original image to the new image using the combined CDF
    # Initialize the output image
    img_equalized = np.zeros_like(img, dtype=np.uint16)

    # Apply the mapping for low and high parts of the image
    img_equalized[mask_low] = np.interp(img[mask_low], bins[:-1], cdf_low)
    img_equalized[mask_high] = np.interp(img[mask_high], bins[:-1], cdf_high)

    return img_equalized


def piecewise_linear_transformation(
    img, low_peak=25000, high_peak=65535 // 2, low_mapped=20000, high_mapped=40000
):
    # Define control points for piecewise linear transformation
    original_values = [0, low_peak, high_peak, 65535]
    target_values = [0, low_mapped, high_mapped, 65535]

    # Perform the piecewise linear transformation
    img_transformed = np.interp(img.flatten(), original_values, target_values)

    # Reshape the 1D array back to a 2D image
    img_transformed = img_transformed.reshape(img.shape)

    return img_transformed.astype(np.uint16)
