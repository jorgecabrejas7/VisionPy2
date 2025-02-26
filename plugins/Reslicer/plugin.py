# Import necessary modules
from base_plugin import BasePlugin
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import matplotlib.pyplot as plt
import tifffile as tiff
from utils import image_sequence
import os
import plugins.Reslicer.reslicer as rsl
# Define a class that implements the PluginInterface
class Plugin(BasePlugin):
    # Prompt the user to select a file
    def run(self):
        file_load = self.request_gui(self.ask_folder_file_save, False)

        if file_load:
            # Open a file dialog to select a file
            file_path = self.select_file("Select File")
        else:
            file_path = self.select_folder("Select Folder")

        file_save = self.request_gui(self.ask_folder_file_save, True)

        if file_save:
            # Open a file dialog to select a folder to save the rotated volume
            save_path = self.select_save_file("Select file to save the resliced volume")
        else:
            save_path = self.select_folder("Select folder to save the resliced volume")

        if file_load:
            # load the file
            volume = tiff.imread(file_path)
        else:
            # load the file
            volume = image_sequence.read_sequence2(file_path, progress_window=self)

        self.update_progress(100, "Volume loaded successfully.")

        # Check if a file was selected
        if file_path and save_path:

            #ask for reslice direction
            reslice_direction = self.request_gui(rsl.ask_reslice,self)

            #reslice the volume
            resliced = rsl.reslice(volume, reslice_direction)

            resliced = self.request_gui(rsl.ask_rotate,self,resliced)

            print(resliced.shape)

            # save the resliced volume
            if file_save:
                self.update_progress(0, "Saving resliced volume.")
                tiff.imwrite(save_path, resliced)
            else:
                image_sequence.write_sequence2(
                    save_path,
                    os.path.basename(save_path),
                    resliced,
                    progress_window=self,
                )
            self.update_progress(100, "Resliced volume saved successfully.")
            self.prompt_message("Resliced volume saved successfully.")
            self.request_gui(plt.close, "all")

        else:
            # Show a message box if no file is selected
            self.prompt_error("No file selected.")

 