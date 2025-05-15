# Import necessary modules
from base_plugin import BasePlugin
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from utils import image_sequence
import os
import numpy as np
import matplotlib.pyplot as plt
import plugins.Centering.centering as cnt
from preprocess_tools import reslicer
from preprocess_tools import aligner
import logging
import tifffile as tiff


# Define a class that implements the PluginInterface
class Plugin(BasePlugin):
    # Prompt the user to select a file
    def execute(self):
        """
        Prompt the user to select a file and process the selected file.

        Raises:
            FileNotFoundError: If no file is selected.

        Example:
            run()
        """

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
        
        # Check if a file was selected
        if file_path and save_path:

            if file_load:
                # load the file
                volume = tiff.imread(file_path)
            else:
                # load the file
                volume = image_sequence.read_sequence2(file_path, progress_window=self)

            #######ALIGNING######
            self.update_progress(0, "Aligning volume.")
            volume = aligner.main(volume,False)
            
            #reslice and rotate90

            volume = reslicer.rotate_auto(volume)
            volume = reslicer.reslice(volume,'Bottom')

            # save the resliced volume
            if file_save:
                self.update_progress(0, "Saving resliced volume.")
                tiff.imwrite(save_path, volume)
            else:
                image_sequence.write_sequence2(
                    save_path,
                    os.path.basename(save_path),
                    volume,
                    progress_window=self,
                )
            self.update_progress(100, "Resliced volume saved successfully.")
            self.prompt_message("Resliced volume saved successfully.")
            self.request_gui(plt.close, "all")

        else:
            self.prompt_message("No file is selected.")

        return

    