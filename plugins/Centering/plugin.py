# Import necessary modules
from base_plugin import BasePlugin
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from utils import image_sequence
import os
import matplotlib.pyplot as plt
import plugins.Centering.centering as cnt


# Define a class that implements the PluginInterface
class Plugin(BasePlugin):
    # Prompt the user to select a file
    def run(self):
        """
        Prompt the user to select a file and process the selected file.

        Raises:
            FileNotFoundError: If no file is selected.

        Example:
            run()
        """

        # Open a file dialog to select a file
        file_path = self.select_folder("Select Folder")

        # Open a file dialog to select a folder to save the centered volume
        save_path = self.select_folder("Select Folder to save centered volume")

        # Check if a file was selected
        if file_path:
            print("the path is: ", file_path)

            # Create a progress window

            self.update_progress(0, "Loading Volume")

            volume = image_sequence.read_sequence2(file_path, progress_window=self)

            self.update_progress(100, "Loading Volume", 1, 1)

            # Select thresholds
            user_inputs = self.request_gui(cnt.get_thresholds, self, volume)

            # Perform operations on the selected file based on user inputs
            if len(user_inputs) > 0:
                # main
                # Create a progress window

                self.update_progress(0, "Centering Volume")
                # get shift
                shift = user_inputs["Main"]
                # shift volume
                volume = cnt.shift_volume_concurrent(
                    volume, shift, progress_window=self
                )

                # select dimensions for cropping
                user_inputs2 = self.request_gui(cnt.get_dimensions, self, volume)

                if user_inputs2:
                    # cropping

                    # cropping

                    volume = cnt.crop_volume(
                        volume, user_inputs2["Main"][0], user_inputs2["Main"][1]
                    )

                    # Check if a folder was selected
                    if save_path:
                        # Create a progress window

                        # Save the centered volume
                        self.update_progress(0, "Saving centered Volume")
                        image_sequence.write_sequence2(
                            save_path,
                            os.path.basename(save_path),
                            volume,
                            progress_window=self,
                        )

                        # Show a message box if the centered volume is saved successfully, when the message box is closed, close all matplotlib figures
                        self.prompt_message("centered volume saved successfully.")
                        self.request_gui(plt.close, "all")

                        self.update_progress(100, "Saving centered Volume", 1, 1)

            else:
                # Show a message box if user inputs are not provided
                self.prompt_error("No input detected.")

        else:
            self.prompt_error("No file selected.")

        return 0
