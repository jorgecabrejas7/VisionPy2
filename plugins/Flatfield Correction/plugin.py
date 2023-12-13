# plugins/Flatfield%20Correction/plugin.py

import os
import sys
import time
import traceback
from pathlib import Path
from typing import *

from base_plugin import BasePlugin
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from pystackreg.util import to_uint16

from utils.bit_depth import f32_to_uint16
from utils.image_sequence import read_virtual_sequence
from utils.image_utils import read_tif
from utils.progress_window import ProgressWindow
from utils.gui_utils import virtual_sequence_bbox


class Plugin(BasePlugin):
    def __init__(self, main_window):
        super().__init__(main_window)

    def execute(self):
        pass