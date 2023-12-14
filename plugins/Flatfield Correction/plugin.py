# plugins/Flatfield%20Correction/plugin.py

import os
import sys
import time
import traceback
from pathlib import Path
from typing import *

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from pystackreg.util import to_uint16

from base_plugin import BasePlugin
from utils.bit_depth import f32_to_uint16
from utils.gui_utils import virtual_sequence_bbox
from utils.image_sequence import read_virtual_sequence
from utils.image_utils import read_tif
from utils.progress_window import ProgressWindow


class Plugin(BasePlugin):
    def __init__(self, main_window, plugin_name):
        super().__init__(main_window, plugin_name)

    def execute(self):
        folder = self.select_folder(caption="Folder testing")
        file = self.select_file(caption="File testing")
        volume = read_virtual_sequence(folder)
        slice_n, bbox = self.get_volume_bbox(volume)
        print(f"{folder = } {file = } {slice_n = } {bbox = }")
