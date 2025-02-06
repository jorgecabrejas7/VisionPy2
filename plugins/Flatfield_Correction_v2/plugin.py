# plugins/Flatfield%20Correction/plugin.py
import logging
import math
import os
import traceback
from concurrent.futures import ThreadPoolExecutor

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from pystackreg.util import to_uint16
import threading
import queue

from base_plugin import BasePlugin
from utils.bit_depth import f32_to_uint16
from utils.image_sequence import read_virtual_sequence
from utils.image_utils import read_tif
from utils.register import stackreg_translate
from utils.gui_utils import get_bbox


class Plugin(BasePlugin):
    def __init__(self, main_window, plugin_name, uuid):
        super().__init__(main_window, plugin_name, uuid)
        self.processing_queue = queue.Queue()
        self.saving_queue = queue.Queue()
        self.stop_event = threading.Event()

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

            self.volume = read_virtual_sequence(folder)
            self.n_slices = self.volume.shape[0]
            slice_n, bbox = self.get_volume_bbox(self.volume)
            # _, self.cropping_bbox = self.get_volume_bbox(self.volume)
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
            self.rotate = self.ask_for_confirmation(
                "Want to rotate volume 180 degrees?"
            )
            x1, y1, x2, y2 = bbox
            flat_roi = flatfield_copy[y1:y2, x1:x2].copy()
            vol_roi = np.array(self.volume[slice_n, y1:y2, x1:x2]).copy()

            sr = stackreg_translate(vol_roi, flat_roi)
            # Apply the translation to the flatfield
            flatfield_copy = sr.transform(flatfield_copy)
            flatfield_copy = to_uint16(flatfield_copy)

            # Copy the detector defects back to the corrected and translated flatfield
            inverted_ddm = (255 - ddm).astype(bool)  # Convert to boolean mask
            flatfield_copy[inverted_ddm] = flatfield[inverted_ddm]

            # Get folder to save the data
            self.save_folder = self.select_folder(
                caption="Select folder to save results"
            )

            # Add correction for loading partial volume (centered), transpose, show, check line, calculate angle and get everything inside the for loop  into a function and paralellize with dask

            if not self.save_folder:
                self.prompt_error("No folder selected")
                self.finished.emit(self)
                return

            right_side = self.load_partial_volume(folder)

            slice_idx = right_side.shape[0] // 2
            show_slice = right_side[slice_idx]
            line_coords = self.request_gui(select_angle_line, show_slice)
            print(line_coords)
            line_coords = [
                (0, line_coords[0][1]),
                (self.volume.shape[0], line_coords[1][1]),
            ]

            dx = line_coords[1][0] - line_coords[0][0]
            dy = line_coords[1][1] - line_coords[0][1]

            # Calculate the angle in radians
            angle_radians = abs(np.arctan2(dy, dx))
            logging.debug(f"Correction angle -> {np.degrees(angle_radians)}")

            # Determine the direction of the line
            self.direction = (
                "upwards" if dy < 0 else "downwards" if dy > 0 else "horizontal"
            )
            representative_slice = (
                self.volume[0] if self.direction == "upwards" else self.volume[-1]
            )
            shift_size = math.ceil(
                2 * self.volume.shape[0] * math.sin(angle_radians / 2)
            )
            representative_slice = np.roll(representative_slice, -shift_size, axis=0)
            self.cropping_bbox = self.request_gui(get_bbox, representative_slice)
            x1, y1, x2, y2 = self.cropping_bbox
            _ = self.request_gui(get_bbox, representative_slice[y1:y2, x1:x2])
            iterator = range(self.volume.shape[0])

            num_processing_threads = 6
            num_saving_threads = 2
            flatfield_copy = flatfield_copy.astype(np.float32) / np.mean(flatfield_copy)
            for _ in range(num_processing_threads):
                processing_thread = threading.Thread(
                    target=self.processing_worker,
                    args=(
                        self.processing_queue,
                        self.saving_queue,
                        flatfield_copy,
                        angle_radians,
                        self.save_folder,
                    ),
                )
                processing_thread.daemon = True
                processing_thread.start()

            for _ in range(num_saving_threads):
                saving_thread = threading.Thread(
                    target=self.saving_worker, args=(self.saving_queue,)
                )
                saving_thread.daemon = True
                saving_thread.start()

            for index in iterator:
                self.processing_queue.put((index, original_slice_names[index]))

            self.processing_queue.join()
            self.saving_queue.join()

            self.stop_event.set()
            self.update_progress(100, "Finished...", self.n_slices, self.n_slices)

        except Exception as e:
            traceback.print_exc()
            self.prompt_error(f"Error while running Flatfield Correction: {str(e)}")
            self.finished.emit(self)
            return

    def correct_slice(self, index, _slice, flatfield_copy, angle_radians):
        """
        Corrects a single slice based on the flatfield copy and the angle.

        Args:
            index (int): The index of the slice in the volume.
            _slice (np.array): The slice to be corrected.
            flatfield_copy (np.array): The flatfield copy used for correction.
            angle_radians (float): The angle in radians used for the shift calculation.

        Returns:
            np.array: The corrected slice.
        """
        # Perform element-wise division of _slice by flatfield_copy
        corrected_slice = np.divide(
            _slice.astype(np.float32),
            flatfield_copy.astype(np.float32),
            out=np.zeros_like(_slice, dtype=np.float32),
            where=flatfield_copy.astype(np.float32) != 0,
        )

        # Calculate the shift size based on the angle and apply it
        correction_index = (
            index if self.direction == "downwards" else self.n_slices - index
        )

        shift_size = math.ceil(2 * correction_index * math.sin(angle_radians / 2))
        shifted_slice = np.roll(corrected_slice, -shift_size, axis=0)

        # Convert the result to uint16 format with scaling
        corrected_slice_uint16 = f32_to_uint16(
            shifted_slice, do_scaling=True, bbox=self.cropping_bbox
        )
        logging.debug(
            f"Flatfield corrected for slice {index} with shift = 2 x {correction_index} x sin({angle_radians} / 2) = {shift_size} pixels - Angle: {np.degrees(angle_radians)}"
        )

        return corrected_slice_uint16

    def load_partial_volume(self, folder):
        files = sorted(
            [
                file
                for file in os.listdir(folder)
                if file.endswith(".tif") or file.endswith(".tiff")
            ]
        )
        files = files[::100]
        index_files = enumerate(files)

        # Concurrently load each of the files as a numpy array that will later be stacked as a 3D volume

        def load_file(tup):
            index, file = tup
            with tifffile.TiffFile(os.path.join(folder, file)) as tif:
                array = tif.asarray(out="memmap")
                new_width = array.shape[1] // 100
                if index % 100 == 0:
                    logging.info(f"Loading partial volume - {file = }")
                return array[
                    :, array.shape[1] // 2 : array.shape[1] // 2 + new_width
                ].copy()

        with ThreadPoolExecutor() as executor:
            arrays = list(executor.map(load_file, index_files))
        logging.info("Finished loading partial volume")
        volume = np.stack(arrays, axis=0)
        return volume.transpose(2, 1, 0)

    def processing_worker(
        self, processing_queue, saving_queue, flatfield_copy, angle_radians, save_folder
    ):
        while not self.stop_event.is_set():
            try:
                index, name = processing_queue.get(timeout=1)

                # Load the slice just before processing
                slice_data = self.volume[index]
                corrected_slice = self.correct_slice(
                    index, slice_data, flatfield_copy, angle_radians
                )
                del slice_data  # Free memory of the loaded slice

                saving_queue.put((index, corrected_slice, name))
                processing_queue.task_done()
            except queue.Empty:
                continue

    def saving_worker(self, saving_queue):
        while not self.stop_event.is_set():
            try:
                index, corrected_slice, name = saving_queue.get(
                    timeout=1
                )  # Timeout to allow checking stop_event
                self.save_slice(index, corrected_slice, name, self.save_folder)
                saving_queue.task_done()
                self.update_progress_after_saving(index)
            except queue.Empty:
                continue

    def update_progress_after_saving(self, index):
        # Update progress after saving
        self.update_progress(
            int((index * 100) / self.n_slices),
            f"Processing slice {index + 1}",
            index,
            self.n_slices,
        )

    def save_slice(self, index, corrected_slice, name, save_folder):
        try:
            x1, y1, x2, y2 = self.cropping_bbox
            _corrected_slice = corrected_slice[y1:y2, x1:x2]
            if self.rotate:
                _corrected_slice = np.rot90(_corrected_slice, 2)
            save_path = os.path.join(save_folder, name)
            tifffile.imwrite(save_path, _corrected_slice)
            logging.debug(f"Slice saved: {save_path}")
        except Exception as e:
            logging.error(f"Failed to save slice {index}: {e}")


def select_angle_line(vol):
    fig, ax = plt.subplots()
    ax.imshow(vol, cmap="gray")
    line_coords_resized = None  # Initialize to None to store only the last line
    current_line = (
        None  # To store the current line object for removal if a new line is drawn
    )

    def on_click(event):
        nonlocal line_coords_resized, current_line
        if fig.canvas.toolbar.mode == "":
            if line_coords_resized is None:
                # If no point is selected yet, store the first point
                line_coords_resized = [(event.xdata, event.ydata)]
            elif len(line_coords_resized) == 1:
                # Store the second point and draw the line
                line_coords_resized.append((event.xdata, event.ydata))
                if current_line:
                    current_line.remove()
                (current_line,) = ax.plot(
                    [line_coords_resized[0][0], line_coords_resized[1][0]],
                    [line_coords_resized[0][1], line_coords_resized[1][1]],
                    "r-",
                )
                fig.canvas.draw()
            else:
                # If a line is already drawn, reset the first point
                line_coords_resized = [(event.xdata, event.ydata)]

    def capture_line_and_return():
        dlg = QDialog()
        dlg.setWindowTitle("Confirm Line Selection")
        layout = QVBoxLayout()

        line_label = QLabel("Select a line by clicking two points on the image.")
        layout.addWidget(line_label)

        if line_coords_resized and len(line_coords_resized) == 2:
            line_text = f"Selected Line: Start({line_coords_resized[0]}), End({line_coords_resized[1]})"
            line_label.setText(line_text)

        ok_button = QPushButton("OK", dlg)
        ok_button.clicked.connect(lambda: (dlg.accept(), plt.close(fig)))
        layout.addWidget(ok_button)

        dlg.setLayout(layout)
        dlg.show()
        plt.show(block=False)
        dlg.exec()

        return line_coords_resized

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    line_coords = capture_line_and_return()
    fig.canvas.mpl_disconnect(cid)
    return line_coords
