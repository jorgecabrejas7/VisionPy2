# Import necessary modules
from base_plugin import BasePlugin
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import matplotlib.pyplot as plt
import tifffile as tiff
import numpy as np
from preprocess_tools import reslicer, aligner, onlypores

# Define a class that implements the PluginInterface
class Plugin(BasePlugin):
    # Prompt the user to select a file
    def run(self):
        # Ask if sample_mask and binary should be saved
        save_masks = self.prompt_confirmation("Do you want to save the sample_mask and binary?")

        batchmode = True
        files_paths = []

        while batchmode:

            # Open a file dialog to select a file
            files_paths.append(self.select_file("Select XCT File"))

            batchmode = self.prompt_confirmation("Do you want to use more files?")
        
        for index, file_path in enumerate(files_paths):

            self.update_progress(0, "Processing "+ file_path, index, len(files_paths))

            # Check if a file was selected
            if file_path:

                # Create a progress window
                self.update_progress(0, "Loading "+ file_path, index, len(files_paths))
                # Load the 3D volume from a .tif file
                volume = tiff.imread(file_path)
                #crop frontwal and backwall
                self.update_progress(25, "Cropping "+ file_path, index, len(files_paths))
                #axes of volume are (z, y, x)
                #transform the volume to (x, y, z)
                volume = np.transpose(volume, (2, 1, 0))
                #axes of volume are (x, y, z)
                #crop the volume to remove the walls
                _,frontwall,backwall = aligner.crop_walls(volume)
                #axes of cropped_volume are (x, y, z)
                #transform the volume to (z, y, x)
                volume = np.transpose(volume, (2, 1, 0))
                # Apply the onlypores function to the volume
                self.update_progress(50, "Calculating "+ file_path, index, len(files_paths))
                onlypores_volume, sample_mask, binary = onlypores.onlypores(volume,frontwall,backwall,sauvola_radius = 49, sauvola_k = 0.05)
                self.update_progress(75, "Saving "+ file_path, index, len(files_paths))
                save_path = file_path.replace(".tif", "_onlypores.tif")
                tiff.imsave(save_path, onlypores_volume.astype(np.uint8) * 255)

                # Save sample_mask and binary if requested
                if save_masks:
                    sample_mask_path = file_path.replace(".tif", "_sample_mask.tif")
                    tiff.imsave(sample_mask_path, sample_mask.astype(np.uint8) * 255)
                    binary_path = file_path.replace(".tif", "_binary.tif")
                    tiff.imsave(binary_path, binary.astype(np.uint8) * 255)

            else:
                # Show a message box if no file is selected
                self.prompt_error("No file selected.")
        self.update_progress(100, "Processed", 1, 1)
        self.prompt_message("All files have been processed.")
