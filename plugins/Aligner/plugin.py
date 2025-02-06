# Import necessary modules
from base_plugin import BasePlugin
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from utils import image_sequence
import os
import numpy as np
import matplotlib.pyplot as plt
from plugins.Aligner import aligner as alg
import logging


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

        # Open a file dialog to select a file
        file_path = self.select_folder("Select Folder")

        # Open a file dialog to select a folder to save the rotated volume
        save_path = self.select_folder("Select Folder to save rotated volume")

        # Check if a file was selected
        if file_path:
            print("the path is: ", file_path)

            # Create a progress window
            self.update_progress(0, "Loading Volume")

            volume = image_sequence.read_sequence2(file_path, progress_window=self)

            # close the progress window
            self.update_progress(100, "Loading Volume", 1, 1)

            reslices = self.request_gui(alg.get_user_inputs3)

            # Select thresholds
            angles = self.request_gui(
                alg.get_angles, plugin=self, volume=volume, reslices=reslices
            )

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

                # Check if a folder was selected
                if save_path:
                    # Create a progress window
                    # Save the rotated volume
                    self.update_progress(0, "Saving Rotated Volume")
                    image_sequence.write_sequence2(
                        save_path,
                        os.path.basename(save_path),
                        volume,
                        progress_window=self,
                    )

                    # Show a message box if the rotated volume is saved successfully, when the message box is closed, close all matplotlib figures
                    self.prompt_message("The rotated volume is saved successfully.")
                    self.request_gui(plt.close, "all")
                    self.update_progress(100, "Saving Rotated Volume", 1, 1)

                else:
                    # Show a message box if no folder is selected
                    self.prompt_message("No saving folder is selected.")

            else:
                # Show a message box if user inputs are not provided
                self.prompt_message("User inputs are not provided.")

        else:
            self.prompt_message("No file is selected.")

        return
