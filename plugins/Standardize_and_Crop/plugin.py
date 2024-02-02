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
    def __init__(self, main_window, plugin_name, uuid):
        super().__init__(main_window, plugin_name, uuid)
        self.processing_queue = queue.Queue()
        self.saving_queue = queue.Queue()
        self.total_slices = 0
        self.stop_event = threading.Event()

    def execute(self):
        try:
            eq_settings = self.request_gui(
                create_equalization_settings_dialog(self.main_window)
            )
            self.target_mean = eq_settings["target_mean"]
            self.target_std = eq_settings["target_std"]
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
            _, self.bbox = self.get_volume_bbox(self.volume)

            # Get folder to save the data
            self.save_folder = self.select_folder(
                caption="Select folder to save results"
            )

            # Add correction for loading partial volume (centered), transpose, show, check line, calculate angle and get everything inside the for loop  into a function and paralellize with dask

            if not self.save_folder:
                self.prompt_error("No folder selected")
                self.finished.emit(self)
                return

            

            # Start processing and saving threads
            num_processing_threads = 8  # You can adjust this value
            num_saving_threads = 4
            for _ in range(num_processing_threads):
                processing_thread = threading.Thread(
                    target=self.processing_worker, 
                    args=(self.processing_queue, self.saving_queue)
                )
                processing_thread.daemon = True
                processing_thread.start()

            
            for _ in range(num_saving_threads):
                saving_thread = threading.Thread(target=self.saving_worker, args=(self.saving_queue,))
                saving_thread.daemon = True
                saving_thread.start()

            for i in range(self.n_slices):
                # Enqueue the slice for processing
                self.processing_queue.put((i, original_slice_names[i]))
            
            # Wait for all tasks to be processed
            self.processing_queue.join()
            self.saving_queue.join()

            # Signal the threads to stop
            self.stop_event.set()

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
    
    def processing_worker(self, processing_queue, saving_queue):
            while not self.stop_event.is_set():
                try:
                    x1, y1, x2, y2 = self.bbox
                    i, name = processing_queue.get(timeout=1)
                    slice = self.volume[i, y1:y2, x1:x2]
                    standardized_slice = self.standardize_slice(slice, self.target_mean, self.target_std)
                    slice_8bit = convert_to_8bit(standardized_slice)
                    eq_slice = np.empty_like(slice_8bit)
                    cv2.intensity_transform.autoscaling(slice_8bit, eq_slice)
                    
                    # Put the processed slice into the saving queue
                    saving_queue.put((i, eq_slice, name))
                    del slice, standardized_slice, slice_8bit, eq_slice
                    processing_queue.task_done()
                except queue.Empty:
                    continue
    def saving_worker(self, saving_queue):
        while not self.stop_event.is_set():
            try:
                i, eq_slice, name = saving_queue.get(timeout=1)
                # Save the cropped slice as a page in the TIFF file
                tifffile.imwrite(
                    os.path.join(self.save_folder, "cropped_" + name),
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
                saving_queue.task_done()
            except queue.Empty:
                continue

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


