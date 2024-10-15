# Import necessary modules
from base_plugin import BasePlugin
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import matplotlib.pyplot as plt
import tifffile as tiff
import plugins.AutoUTAligner.autoutaligner as autoalg


# Define a class that implements the PluginInterface
class Plugin(BasePlugin):
    # Prompt the user to select a file
    def run(self):

        batchmode = True

        files_paths = []

        while batchmode:

            # Open a file dialog to select a file
            files_paths.append(self.select_file("Select UT File"))

            batchmode = self.prompt_confirmation("Do you want to use more files?")
        
        for file_path in files_paths:

            # Check if a file was selected
            if file_path:

                # Create a progress window

                # Load the 3D volume from a .tif file
                volume = tiff.imread(file_path)
                # Apply the Hilbert transform to the volume
                amplitude= autoalg.hillbert_transform(volume)
                save_path = file_path.replace(".tif", "_amplitude.tif")
                tiff.imsave(save_path, amplitude)

            else:
                # Show a message box if no file is selected
                self.prompt_error("No file selected.")
        self.update_progress(100, "Aligned", 1, 1)
        self.prompt_message("All files have been aligned.")

    