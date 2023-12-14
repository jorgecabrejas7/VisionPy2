# base_plugin.py

import traceback
from abc import abstractmethod
from typing import *

import zarr
from PyQt6.QtCore import QEventLoop, QObject, pyqtSignal
from PyQt6.QtWidgets import QFileDialog

from utils.gui_utils import virtual_sequence_bbox


class BasePlugin(QObject):
    finished = pyqtSignal(object)  # Signal to emit results or status
    error = pyqtSignal(str)  # Signal to emit error messages
    progress = pyqtSignal(
        [int], [int, str, int, int]
    )  # Signal to emit progress updates

    #  Signals for other GUI requests
    request_gui_interaction = pyqtSignal(
        object, tuple, dict
    )  # Callback function that should return a value and kwargs dict
    # All arguments passed to the callback are keyword arguments

    request_stop = pyqtSignal()  # Signal to request the plugin to stop

    """
    Create a common way request_ui() that sends a signal with a callback that returns the result
    of the gui interaction. This will be used to request a file or folder from the user, for example.
    Then, the main thread (main window) will send the requested_gui signal with the result, storing it
    on an instance variable to later be returned by the wrapper function for such interaction    
    """

    @abstractmethod
    def __init__(self, main_window, plugin_name):
        super().__init__()
        self.gui_result = None
        self.main_window = main_window
        self.loop = None
        self.name = plugin_name

    @abstractmethod
    def get_name(self):
        """
        Return the name of the plugin.

        This method should be implemented to return a unique name
        for the plugin.
        """
        return self.name

    @abstractmethod
    def get_description(self):
        """
        Return a description of the plugin.

        This method should be implemented to provide a brief description
        of what the plugin does.
        """
        pass

    def run(self, *args, **kwargs):
        """
        A wrapper method to run the image processing in a separate thread.

        Parameters:
            image: The image to be processed.
        """
        try:
            result = self.execute(*args, **kwargs)
            self.finished.emit(result)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
            self.request_stop.emit()

    """
        Main functions for the GUI request mechanism. Some of the more common ones will be implemented here.
        Any other kind of GUI interaction should be implemented in the plugin itself, using the provided approach 
        aligning with Qt signal-slot mechanism. 

        Callback functions must implement the logic of the GUI and return a value, even if it is a boolean meaning
        error or success. For ease of use and readability, functions such as selecting a file define the callback
        inside themselves. All they have to do is return request_gui(calllback, kwargs) and the result will be returned
        as if it was a normal function call.
    """

    def request_gui(self, callback, *args, **kwargs):
        self.loop = QEventLoop()
        self.gui_result = None

        # connect to the signal
        self.main_window.gui_response.connect(self.on_gui_response)
        self.request_gui_interaction.emit(callback, args, kwargs)
        self.loop.exec()
        # disconnect after the signal is received to allow other plugins to use the signal
        self.main_window.gui_response.disconnect(self.on_gui_response)
        return self.gui_result

    def on_gui_response(self, result):
        self.gui_result = result
        self.loop.exit()

    def select_file(self, caption: str = None):
        def callback(caption):
            file, _ = QFileDialog.getOpenFileName(self.main_window, caption=caption)
            return file

        return self.request_gui(
            callback, caption=caption if caption else "Select a file"
        )

    def select_folder(self, caption: str = None):
        def callback(caption):
            folder = QFileDialog.getExistingDirectory(self.main_window, caption=caption)
            return folder

        return self.request_gui(
            callback, caption=caption if caption else "Select a folder"
        )

    def get_volume_bbox(self, zarr_array: zarr.Array) -> Tuple[int, List[int]]:
        """
        Processes a Zarr array, displaying each slice and allowing the user to select a
        bounding box on a specific slice. Captures the details via a dialog box.

        Args:
        zarr_array (zarr.Array): Zarr array representing image slices.

        Returns:
        Tuple[int, List[int]]: Tuple containing the index of the selected slice and the bounding box coordinates.
        """
        return self.request_gui(virtual_sequence_bbox, zarr_array=zarr_array)
