import importlib.util
import os
import subprocess
import sys
import traceback

import pkg_resources
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QDialog,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from utils.gui_utils import *

from utils.progress_window import ProgressWindow


class ProgressDialog(QDialog):
    """
    A dialog window that displays a progress bar for installing plugin dependencies.

    Args:
        total (int): The total number of steps in the installation process.

    Attributes:
        layout (QVBoxLayout): The layout of the dialog window.
        progressBar (QProgressBar): The progress bar widget.

    Methods:
        update_progress(value): Updates the progress bar with the specified value.
    """

    def __init__(self, total):
        super().__init__()
        self.setWindowTitle("Installing Plugin Dependencies")
        self.layout = QVBoxLayout(self)
        self.progressBar = QProgressBar(self)
        self.progressBar.setMaximum(total)
        self.layout.addWidget(self.progressBar)

    def update_progress(self, value):
        """
        Updates the progress bar with the specified value.

        Args:
            value (int): The current progress value.
        """
        self.progressBar.setValue(value)


class InstallThread(QThread):
    """
        A thread class for installing requirements.

        This class extends the QThread class and provides functionality for installing
        a list of requirements in a separate thread. It emits signals to indicate the
        progress of the installation and any errors that occur.

        Attributes:
            progress (pyqtSignal): A signal emitted to indicate the progress of the installation.
                The signal carries an integer value representing the current progress.
            error (pyqtSignal): A signal emitted when an error occurs during the installation.
                The signal carries a string value representing the error message.
    # InstallThread definition
            requirements (list): A list of requirements to be installed.

        Methods:
            run(): The main method of the thread. It performs the installation logic and emits
                signals to indicate the progress and errors.

    """

    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, requirements):
        super().__init__()
        self.requirements = requirements

    def run(self):
        for i, requirement in enumerate(self.requirements, start=1):
            try:
                pkg_resources.require(requirement)
            except pkg_resources.DistributionNotFound:
                try:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", requirement]
                    )volume_utils
                except subprocess.CalledProcessError as e:
                    self.error.emit(f"Failed to install {requirement}: {e}")
                    break
            except pkg_resources.VersionConflict as e:
                self.error.emit(f"Version conflict for {requirement}: {e}")
                break
            except Exception as e:
                self.error.emit(str(e))
                break
            else:
                self.progress.emit(i)


# MainWindow definition
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.progress_windows = {}
        self.threads = {}
        self.init_ui()
        self.discover_plugins()

    def init_ui(self):
        self.setWindowTitle("VisionPy")
        self.setGeometry(300, 300, 400, 100)
        self.plugin_menu = self.menuBar().addMenu("Plugins")
        self.plugin_toolbar = QToolBar("Plugin Toolbar")
        self.addToolBar(self.plugin_toolbar)
        reload_plugins_action = QAction("Reload Plugins", self)
        reload_plugins_action.triggered.connect(self.discover_plugins)
        self.plugin_toolbar.addAction(reload_plugins_action)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

    def discover_plugins(self):
        plugin_dir = os.path.join(os.getcwd(), "plugins")
        if not os.path.exists(plugin_dir):
            return

        self.plugin_menu.clear()
        self.plugins = {}
        requirements = []

        for plugin_name in os.listdir(plugin_dir):
            plugin_folder_path = os.path.join(plugin_dir, plugin_name)
            if os.path.isdir(plugin_folder_path):
                requirements_path = os.path.join(plugin_folder_path, "requirements.txt")
                if os.path.isfile(requirements_path):
                    with open(requirements_path, "r") as reqs:
                        for requirement in reqs:
                            requirement = requirement.strip()
                            if requirement:
                                requirements.append(requirement)
                plugin_file_path = os.path.join(plugin_folder_path, "plugin.py")
                if os.path.isfile(plugin_file_path):
                    self.import_plugin(plugin_name, plugin_file_path)

        if requirements:
            self.show_installation_dialog(requirements)

    def show_installation_dialog(self, requirements):
        dialog = ProgressDialog(len(requirements))
        thread = InstallThread(requirements)
        thread.progress.connect(dialog.update_progress)
        thread.error.connect(
            lambda e: QMessageBox.critical(self, "Installation Error", e)
        )
        thread.finished.connect(dialog.accept)
        thread.start()
        dialog.exec()

    def import_plugin(self, plugin_name, plugin_path):
        spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
        if spec and spec.loader:
            plugin_module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(plugin_module)
                plugin_instance = plugin_module.Plugin(self)
                self.plugins[plugin_name] = plugin_instance

                # Connect signals here if needed
                self.setup_plugin_connections(plugin_instance, plugin_name)

                action = QAction(plugin_name, self)
                action.triggered.connect(
                    lambda checked, p=plugin_instance: self.run_plugin(p)
                )
                self.plugin_menu.addAction(action)
            except Exception as e:
                traceback.print_exc()
                QMessageBox.critical(
                    self, "Plugin Error", f"Failed to load plugin '{plugin_name}': {e}"
                )
                return

    def run_plugin(self, plugin_instance):
        try:
            thread = QThread()
            plugin_name = next(
                (k for k, v in self.plugins.items() if v == plugin_instance), None
            )
            if not plugin_name:
                QMessageBox.critical(self, "Plugin Error", "Failed to find plugin name")
                return

            self.threads[plugin_name] = thread
            plugin_instance.movevolume_utilsToThread(thread)
            plugin_instance.finished.connect(thread.quit)
            thread.started.connect(plugin_instance.execute)
            thread.start()
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(
                self, "Plugin Error", f"An error occurred while running the plugin: {e}"
            )

    def setup_plugin_connections(self, plugin_instance, plugin_name):
        """
        Set up the connections for a plugin instance.

        Args:
            plugin_instance: An instance of the plugin.
            plugin_name: The name of the plugin.

        Returns:
            None
        """
        # Finished signal
        plugin_instance.finished.connect(self.on_plugin_finished)
        # Errir signal
        plugin_instance.error.connect(self.on_plugin_error)
        # Different progress signals - starting one and others
        plugin_instance.progress[int].connect(
            lambda value: self.on_plugin_progress(plugin_name, value)
        )
        plugin_instance.progress[int, str, int, int].connect(
            lambda value, message, index, total: self.on_plugin_progress(
                plugin_name, value, message, index, total
            )
        )
        # Signal to request GUI thread to prompt user for a file
        plugin_instance.request_file_dialog.connect(self.on_file_request)

    # Define the slots for the plugin signals
    def on_plugin_progress(
        self, plugin_name, value, message=None, index=None, total=None
    ):
        if plugin_name not in self.progress_windows:
            self.progress_windows[plugin_name] = ProgressWindow(self)
        if not self.progress_windows[plugin_name].isVisible():
            self.progress_windows[plugin_name].show()
        self.progress_windows[plugin_name].update_progress(value, message, index, total)

    def on_plugin_finished(self, result):
        # Handle the result of the plugin's processing
        pass

    def on_plugin_error(self, error_message):
        # Handle any errors that occurred during the plugin's processing
        QMessageBox.critical(self, "Plugin Error", error_message)

    def on_gui_request(self, caption):
        file, _ = QFileDialog.getOpenFileName(self, caption)
        self.sender().file_selected.emit(file)
