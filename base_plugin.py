# base_plugin.py

import os
import sys
import traceback
import uuid
from abc import abstractmethod

import tifffile as tiff
import zarr
from PyQt6.QtCore import QEventLoop, QObject, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QDialog, QVBoxLayout, QPushButton

from utils import image_sequence
from utils.gui_utils import virtual_sequence_bbox, virtual_sequence_slice
from views.main_window import MainWindow


class BasePlugin(QObject):
    finished = pyqtSignal(object)  # Signal to emit results or status
    error = pyqtSignal(
        str, str
    )  # Signal to emit error messages: (error_message, plugin_name)
    progress = pyqtSignal(int, str, int, int)  # Signal to emit progress updates

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
    def __init__(self, main_window: MainWindow, plugin_name: str, uuid: uuid.UUID):
        super().__init__()
        self.gui_result: object = None
        self.main_window: MainWindow = main_window
        self.loop: QEventLoop = None
        self.name: str = plugin_name
        self.uuid = uuid

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

    @abstractmethod
    def execute(self, *args, **kwargs):
        """
        Execute the plugin.

        This method should be implemented to execute the plugin.
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
        error or success (None is also correct and will be interpreted as False if needed in the process).
        For ease of use and readability, functions such as selecting a file define the callback
        inside themselves. All they have to do is return request_gui(calllback, kwargs) and the result will be returned
        as if it was a normal function call.
    """

    def request_gui(self, callback: object, *args, **kwargs):
        """
        Requests GUI interaction and waits for the response.

        Args:
            callback (object): The callback function to be executed.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The result of the GUI interaction.
        """
        self.loop = QEventLoop()
        self.gui_result = None

        # connect to the signal
        self.main_window.gui_response.connect(self.on_gui_response)
        self.request_gui_interaction.emit(callback, args, kwargs)
        self.loop.exec()
        # disconnect after the signal is received to allow other plugins to use the signal
        self.main_window.gui_response.disconnect(self.on_gui_response)
        return self.gui_result

    def on_gui_response(self, result: object):
        """
        Handles the response from the GUI.

        Args:
            result (object): The result received from the GUI.

        Returns:
            None
        """
        self.gui_result = result
        self.loop.exit()

    def select_file(self, caption: str = None):
        """
        Opens a file dialog to allow the user to select a file.

        Args:
            caption (str, optional): The caption to display in the file dialog. Defaults to None.

        Returns:
            str: The path of the selected file.
        """
        def callback(caption):
            file, _ = QFileDialog.getOpenFileName(self.main_window, caption=caption)
            return file

        return self.request_gui(
            callback, caption=caption if caption else "Select a file"
        )

    def select_save_file(self, caption: str = None):
        """
        Opens a file dialog to select a file for saving.

        Args:
            caption (str, optional): The caption for the file dialog. Defaults to None.

        Returns:
            str: The selected file path.

        """
        def callback(caption):
            file, _ = QFileDialog.getSaveFileName(self.main_window, caption=caption)
            return file

        return self.request_gui(
            callback, caption=caption if caption else "Select a file"
        )

    def select_folder(self, caption: str = None):
            """
            Opens a dialog box to select a folder.

            Args:
                caption (str, optional): The caption for the dialog box. Defaults to None.

            Returns:
                str: The selected folder path.
            """
            def callback(caption):
                if sys.platform == "win32":
                    root_path = "C:\\"
                else:
                    root_path = os.path.expanduser("~")
                folder = QFileDialog.getExistingDirectory(
                    self.main_window, caption, root_path
                )
                return folder

            return self.request_gui(
                callback, caption=caption if caption else "Select a folder"
            )

    def ask_folder_file_save(self, save):
        """
        Opens a dialog window to ask the user to select between saving as a file or a folder.

        Parameters:
        - save (bool): If True, the dialog window is for saving a file. If False, the dialog window is for loading a file.

        Returns:
        - bool: True if the user selects "File", False if the user selects "Folder".
        """
        window = QDialog(self.main_window)
        layout = QVBoxLayout()

        if save:
            window.setWindowTitle("Save format")
        else:
            window.setWindowTitle("Load format")

        button_name = None

        def on_button_clicked(name):
            nonlocal button_name
            button_name = name
            window.close()

        for name in ["File", "Folder"]:
            button = QPushButton(name)
            button.clicked.connect(lambda checked, name=name: on_button_clicked(name))
            layout.addWidget(button)

        window.setLayout(layout)
        window.show()
        window.exec()

        if button_name == "File":
            return True
        else:
            return False

    def load_tiff(self):
        """
        Load a TIFF file or a sequence of TIFF files.

        Returns:
            numpy.ndarray: The loaded TIFF image or sequence of images.
        """
        file_load = self.request_gui(self.ask_folder_file_save, False)

        if file_load:
            # Open a file dialog to select a file
            file_path = self.select_file("Select File")
            return tiff.imread(file_path)
        else:
            file_path = self.select_folder("Select Folder")
            return image_sequence.read_sequence2(file_path, progress_window=self)

    def save_tiff_file(
        self, volume, file_path, imagej=True, metadata={"axes": "ZYX", "unit": "um"}
    ):
        """
        Save a volume as a TIFF file.

        Args:
            volume (ndarray): The volume to be saved.
            file_path (str): The path to save the TIFF file.
            imagej (bool, optional): Whether to save the file in ImageJ format. Defaults to True.
            metadata (dict, optional): Metadata to be included in the TIFF file. Defaults to {'axes': 'ZYX', 'unit': 'um'}.
        """
        tiff.imwrite(file_path, volume, imagej=imagej, metadata=metadata)

    def get_volume_bbox(self, zarr_array: zarr.Array) -> tuple[int, list[int]]:
        """
        Processes a Zarr array, displaying each slice and allowing the user to select a
        bounding box on a specific slice. Captures the details via a dialog box.

        Args:
        zarr_array (zarr.Array): Zarr array representing image slices.

        Returns:
        Tuple[int, List[int]]: Tuple containing the index of the selected slice and the bounding box coordinates.
        """
        return self.request_gui(virtual_sequence_bbox, zarr_array=zarr_array)

    def select_slice(self, zarr_array: zarr.Array) -> int:
        """
        Selects a slice from the given Zarr array.

        Args:
            zarr_array (zarr.Array): The Zarr array from which to select a slice.

        Returns:
            int: The selected slice.

        """
        return self.request_gui(virtual_sequence_slice, zarr_array=zarr_array)

    def update_progress(
        self, value: int, message: str, index: int = None, total: int = None
    ):
        """
        Update the progress bar and label with the given value and message.

        Args:
        - value (int): The value to set the progress bar to.
        - message (str): The message to display in the progress label.

        Raises:
        - No specific exceptions are raised by this function.

        Example:
        ```
        self.update_progress(50, "Processing...")
        ```
        """
        self.progress.emit(value, message, index, total)

    def prompt_error(self, error_caption: str) -> None:
        """
        Displays an error message box with the given error caption.

        Args:
            error_caption (str): The caption for the error message.

        Returns:
            None
        """
        def error_callback(caption: str) -> None:
            QMessageBox.critical(
                self.main_window, "Error", caption, QMessageBox.StandardButton.Ok
            )

        return self.request_gui(error_callback, caption=error_caption)

    # create a function to prompt a message box with a message

    def prompt_message(self, message_caption: str) -> None:
            """
            Displays a message box with the given caption.

            Args:
                message_caption (str): The caption of the message box.

            Returns:
                None
            """
            def message_callback(caption: str) -> None:
                QMessageBox.information(
                    self.main_window, "Message", caption, QMessageBox.StandardButton.Ok
                )

            return self.request_gui(message_callback, caption=message_caption)

    # Create a function to prompot a confirmation box with a message and yes and no buttons, return True if yes and False if no
    def prompt_confirmation(self, message_caption: str) -> bool:
            """
            Prompts the user for confirmation with a message caption.

            Args:
                message_caption (str): The message caption to display.

            Returns:
                bool: True if the user confirms, False otherwise.
            """
            def confirmation_callback(caption: str) -> bool:
                reply = QMessageBox.question(
                    self.main_window,
                    "Confirmation",
                    caption,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                return reply == QMessageBox.StandardButton.Yes

            return self.request_gui(confirmation_callback, caption=message_caption)
