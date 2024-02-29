import importlib.util
import os
import subprocess
import sys
import traceback
import logging
import pkg_resources
import uuid

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
)
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
                    )
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
    gui_response = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.progress_windows = {}
        self.threads = {}
        self.plugin_menu = self.menuBar().addMenu("Plugins")
        self.discover_plugins()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("VisionPy")
        self.setGeometry(300, 300, 400, 100)
        self.plugin_toolbar = QToolBar("Plugin Toolbar")
        self.addToolBar(self.plugin_toolbar)
        reload_plugins_action = QAction("Reload Plugins", self)
        reload_plugins_action.triggered.connect(self.discover_plugins)
        self.plugin_toolbar.addAction(reload_plugins_action)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

    def discover_plugins(self):
        logging.info("Discovering plugins")
        plugin_dir = os.path.join(os.getcwd(), "plugins")
        if not os.path.exists(plugin_dir):
            return

        self.plugin_menu.clear()
        self.plugin_paths = {}
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
                    self.plugin_paths[plugin_name] = plugin_file_path

        for plugin_name in self.plugin_paths:
            display_name = plugin_name.replace("__", ".").replace("_", " ")
            action = QAction(display_name, self)
            action.triggered.connect(
                lambda checked, name=plugin_name: self.run_plugin(name)
            )
            self.plugin_menu.addAction(action)

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

                return plugin_module.Plugin(self, plugin_name, uuid.uuid4())

            except Exception as e:
                traceback.print_exc()
                QMessageBox.critical(
                    self, "Plugin Error", f"Failed to load plugin '{plugin_name}': {e}"
                )

        return None

    def run_plugin(self, plugin_name):
        plugin_path = self.plugin_paths.get(plugin_name)
        if not plugin_path:
            QMessageBox.critical(
                self, "Plugin Error", f"Plugin '{plugin_name}' not found"
            )
            return

        plugin_instance = self.import_plugin(plugin_name, plugin_path)
        logging.info(f"Loaded plugin {plugin_name}")

        if plugin_instance:
            try:
                logging.info(
                    f"Sending plugin {plugin_name} with id {plugin_instance.uuid} to thread"
                )
                thread = QThread()

                self.threads[plugin_instance.uuid] = thread
                plugin_instance.moveToThread(thread)
                self.setup_plugin_connections(plugin_instance, plugin_name)
                thread.started.connect(plugin_instance.run)
                thread.start()
                logging.info(
                    f"Started plugin {plugin_name} with id {plugin_instance.uuid}"
                )
            except Exception as e:
                traceback.print_exc()
                QMessageBox.critical(
                    self,
                    "Plugin Error",
                    f"An error occurred while running the plugin: {e}",
                )

    def cleanup_plugin(self, plugin_name, plugin_instance=None):
        logging.info(f"Cleaning up {plugin_name}")
        thread = self.threads.get(plugin_name)
        id = plugin_instance.uuid
        if thread:
            thread.quit()
            thread.wait()
            thread.deleteLater()
            del self.threads[id]

        try:
            if plugin_instance:
                plugin_instance.deleteLater()
        except Exception:
            logging.info(f"Plugin instance already deleted {plugin_name}")
        logging.info(f"Cleaned {plugin_name}")

    def setup_plugin_connections(self, plugin_instance, plugin_name):
        """
        Set up the connections for a plugin instance.

        Args:
            plugin_instance: An instance of the plugin.
            plugin_name: The name of the plugin.

        Returns:
            None
        """

        # Error signal
        plugin_instance.error.connect(self.on_plugin_error)
        # Different progress signals - starting one and others
        plugin_instance.progress.connect(
            lambda value, message, index, total: self.on_plugin_progress(
                plugin_instance.uuid, value, message, index, total
            )
        )
        # Signal to request GUI thread to prompt user for a file
        plugin_instance.request_gui_interaction.connect(self.on_gui_request)
        thread = self.threads[plugin_instance.uuid]
        plugin_instance.moveToThread(thread)
        plugin_instance.finished.connect(
            lambda: self.cleanup_plugin(plugin_name, plugin_instance=plugin_instance)
        )
        plugin_instance.request_stop.connect(
            lambda: self.cleanup_plugin(plugin_name, plugin_instance=plugin_instance)
        )

    # Define the slots for the plugin signals
    def on_plugin_progress(
        self,
        plugin_id,
        value,
        message=None,
        index=None,
        total=None,  # Signals to receive the gui result
    ):
        if plugin_id not in self.progress_windows:
            self.progress_windows[plugin_id] = ProgressWindow(self)
        if not self.progress_windows[plugin_id].isVisible():
            self.progress_windows[plugin_id].show()

        self.progress_windows[plugin_id].update_progress(
            value,
            message,
            index,
            total,
        )

        if value == 100:
            self.progress_windows[plugin_id].close()
            del self.progress_windows[plugin_id]

    def on_plugin_error(self, error_message, plugin_name):
        # Handle any errors that occurred during the plugin's processing
        traceback.print_exc()
        QMessageBox.critical(self, "Plugin Error", error_message)
        self.cleanup_plugin(plugin_name)

    def on_gui_request(self, callback, args, kwargs):
        try:
            if callback:
                if args and kwargs:
                    result = callback(*args, **kwargs)
                elif args:
                    result = callback(*args)
                elif kwargs:
                    result = callback(**kwargs)
                else:
                    result = callback()

            self.gui_response.emit(result)
        except Exception:
            traceback.print_exc()
            self.gui_response.emit(None)
