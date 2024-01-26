# plugins/Flatfield%20Correction/plugin.py
import logging
import math
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

import cv2
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from dask import compute, delayed
from dask.diagnostics import ProgressBar
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


class Plugin(BasePlugin):
    def __init__(self, main_window, plugin_name):
        super().__init__(main_window, plugin_name)
        self.processing_queue = queue.Queue()
        self.saving_queue = queue.Queue()
        self.total_slices = 0
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

            tiff_path = os.path.join(
                self.save_folder, f"cropped_{os.path.basename(folder)}.tif"
            )
            x1, y1, x2, y2 = bbox

            for i in range(self.n_slices):
                # Read and crop the slice
                slice = self.volume[i, y1:y2, x1:x2]

                # Save the cropped slice as a page in the TIFF file
                tifffile.imwrite(
                    os.path.join(
                        self.save_folder, "cropped_" + original_slice_names[i]
                    ),
                    slice,
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
