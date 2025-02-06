# Import necessary modules
from base_plugin import BasePlugin
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import matplotlib.pyplot as plt
import tifffile as tiff
import plugins.UTAligner.aligner as alg


# Define a class that implements the PluginInterface
class Plugin(BasePlugin):
    # Prompt the user to select a file
    def run(self):
        # Open a file dialog to select a file
        file_path = self.select_file("Select UT File")

        # Open a file dialog to select a folder to save the rotated volume
        save_path = self.select_save_file("Select filename to save the aligned volume")

        # Check if a file was selected
        if file_path and save_path:
            print("the path is: ", file_path)

            # Create a progress window

            volume = tiff.imread(file_path)

            auto = alg.ask_auto(self)

            print(volume.shape)

            if auto:
                gate = alg.autogate(volume)

            else:
                gate = alg.get_gate(self, volume)

            print(gate)

            volume = alg.align(volume, gate)

            # align the x,y axis

            angle = alg.get_angle(volume, gate)

            # create a progress window

            self.update_progress(0, "Rotating Volume")

            rotated_volume = alg.rotate_volume(volume, angle, self)

            self.update_progress(100, "Rotating Volume", 1, 1)

            # Save the rotated volume

            tiff.imsave(save_path, rotated_volume)

            # Show a message box if the rotated volume is saved successfully, when the message box is closed, close all matplotlib figures
            self.prompt_message("Aligned volume saved successfully.")
            self.request_gui(plt.close, "all")

        else:
            # Show a message box if no file is selected
            self.prompt_error("No file selected.")
