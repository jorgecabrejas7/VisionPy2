# base_plugin.py

from abc import ABC, abstractmethod

from PyQt6.QtCore import QObject, pyqtSignal, QEventLoop, QThread
from PyQt6.QtWidgets import QFileDialog


class BasePlugin(QObject):
    finished = pyqtSignal(object)  # Signal to emit results or status
    error = pyqtSignal(str)  # Signal to emit error messages
    progress = pyqtSignal(
        [int], [int, str, int, int]
    )  # Signal to emit progress updates
    #  Signals for dialog requests
    request_file_dialog = pyqtSignal(str)
    request_folder_dialog = pyqtSignal(str)

    # Signals to receive the selected paths
    file_selected = pyqtSignal(str)
    folder_selected = pyqtSignal(str)
    selected_file, selected_folder = None, None

    """
    Create a common way request_ui() that sends a signal with a callback that returns the result
    of the gui interaction. This will be used to request a file or folder from the user, for example.
    Then, the main thread (main window) will send the requested_gui signal with the result, storing it
    on an instance variable to later be returned by the wrapper function for such interaction    
    """

    @abstractmethod
    def __init__(self, main_window):
        super().__init__()
        self.selected_file = None
        self.selected_folder = None
        self.main_window = main_window
        self.loop = None

    @abstractmethod
    def get_name(self):
        """
        Return the name of the plugin.

        This method should be implemented to return a unique name
        for the plugin.
        """
        pass

    @abstractmethod
    def get_description(self):
        """
        Return a description of the plugin.

        This method should be implemented to provide a brief description
        of what the plugin does.
        """
        pass

    @abstractmethod
    def start_processing(self, *args, **kwargs):
        """
        Start processing the data.

        This method should be implemented to start processing the data.
        """
        pass

    def run(self, *args, **kwargs):
        """
        A wrapper method to run the image processing in a separate thread.

        Parameters:
            image: The image to be processed.
        """
        try:
            result = self.start_processing(*args, **kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def ask_for_file(self, caption):
        self.loop = QEventLoop()
        self.selected_file = None

        self.file_selected.connect(self.on_file_received)
        self.request_file_dialog.emit(caption)
        self.loop.exec()
        return self.selected_file

    def on_file_received(self, file):
        self.selected_file = file
        self.loop.exit()
