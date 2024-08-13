# Import necessary modules
from base_plugin import BasePlugin
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from utils import image_sequence
import os
import numpy as np
import matplotlib.pyplot as plt
from plugins.Aligner import aligner as alg
import plugins.Centering.centering as cnt
import plugins.Reslicer.reslicer as rsl
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

        if file_load:
            # load the file
            volume = tiff.imread(file_path)
        else:
            # load the file
            volume = image_sequence.read_sequence2(file_path, progress_window=self)

        # Check if a file was selected
        if file_path:

            reslices = {"Main": True, "Left": True, "Top": True}

            # Select thresholds
            angles = alg.get_angles_auto(plugin = self, volume=volume, reslices=reslices)

            logging.info("-----------------------")

            # Perform operations on the selected file based on user inputs
            if angles:
                if reslices["Main"]:
                    # main
                    # Create a progress window

                    self.update_progress(0, "Rotating Volume from Main reslice")
                    print("The angles: ", angles)
                    angle = angles["Main"]
                    volume = alg.rotate_volume_concurrent(
                        volume, angle, progress_window=self
                    )
                    self.update_progress(100, "Rotating Volume from Main reslice")

                if reslices["Left"]:
                    # Create a progress window
                    self.update_progress(0, "Rotating Volume from Left reslice")
                    volume = np.transpose(volume, (2, 0, 1))
                    angle = angles["Left"]
                    volume = alg.rotate_volume_concurrent(
                        volume, angle, progress_window=self
                    )
                    self.update_progress(100, "Rotating Volume from Left reslice")

                    if not reslices["Top"]:
                        volume = np.transpose(volume, (2, 0, 1))
                        volume = np.transpose(volume, (2, 0, 1))

                if reslices["Top"]:
                    # Create a progress window
                    self.update_progress(0, "Rotating Volume from Top reslice")
                    if not reslices["Left"]:
                        volume = np.transpose(volume, (2, 0, 1))
                    volume = np.transpose(volume, (2, 0, 1))
                    angle = angles["Top"]
                    volume = alg.rotate_volume_concurrent(
                        volume, angle, progress_window=self
                    )
                    self.update_progress(100, "Rotating Volume from Top reslice")

                    volume = np.transpose(volume, (2, 0, 1))

                #######CENTERING######
                
                # Select thresholds
                user_inputs = cnt.get_thresholds_auto(self, volume)

                # Perform operations on the selected file based on user inputs
                if len(user_inputs) > 0:
                    # main
                    # Create a progress window

                    self.update_progress(0, "Centering Volume")
                    #get shift
                    shift = user_inputs['Main']
                    #shift volume
                    volume = cnt.shift_volume_concurrent(volume,shift,progress_window=self)

                    #cropping
                    volume = cnt.crop_volume_auto(volume)
                
                #reslice and rotate90

                #ask for reslice direction
                reslice_direction = "Left"

                #reslice the volume
                resliced = rsl.reslice(volume, reslice_direction)

                resliced = rsl.rotate_auto(resliced)

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
                # Show a message box if user inputs are not provided
                self.prompt_message("User inputs are not provided.")

        else:
            self.prompt_message("No file is selected.")

        return

    