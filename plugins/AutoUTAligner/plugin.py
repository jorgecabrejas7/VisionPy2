# Import necessary modules
from base_plugin import BasePlugin
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import matplotlib.pyplot as plt
import tifffile as tiff
import plugins.UTAligner.aligner as alg
import plugins.AutoUTAligner.autoutaligner as autoalg


# Define a class that implements the PluginInterface
class Plugin(BasePlugin):
    # Prompt the user to select a file
    def run(self):

        batchmode = True

        files_paths = []
        saves_paths = []

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
                
                # Check if the maximum value in the volume is greater than 128
                if autoalg.is_RF(volume):
                    # Apply the Hilbert transform to the volume
                    amplitude= autoalg.hillbert_transform(volume)
                    # Find the range around the brightest slice in the amplitude
                    gate = autoalg.auto_gate(amplitude)
                    print(gate)
                    # Align the amplitude and the volume based on the maximum value in the gated range
                    amplitude, volume = autoalg.double_align(amplitude,volume, gate)
                    # Crop the center of each slice in the amplitude
                    cropped = autoalg.crop_volume(amplitude)
                    # Calculate the rotation angle of the largest component in the cropped amplitude
                    angle = autoalg.angle_max(cropped)
                    # Rotate the volume and the amplitude by the calculated angle
                    rotated_volume = alg.rotate_volume(volume, angle, self)
                    rotated_amplitude = alg.rotate_volume(amplitude, angle, self)
                    # Save the rotated volume and amplitude
                    save_path = file_path.replace(".tif", "_aligned.tif")
                    tiff.imsave(save_path, rotated_volume)
                    save_path = save_path.replace("_aligned.tif", "_amplitude.tif")
                    tiff.imsave(save_path, rotated_amplitude)
                else:
                    # Find the range around the brightest slice in the volume
                    gate = autoalg.auto_gate(volume)
                    print(gate)
                    # Align the volume based on the maximum value in the gated range
                    volume = alg.align(volume, gate)
                    # Crop the center of each slice in the volume
                    cropped = autoalg.crop_volume(volume)
                    # Calculate the rotation angle of the largest component in the cropped volume
                    angle = autoalg.angle_max(cropped)
                    # Rotate the volume by the calculated angle
                    rotated_volume = alg.rotate_volume(volume, angle, self)
                    # Save the rotated volume
                    save_path = file_path.replace(".tif", "_aligned.tif")
                    tiff.imsave(save_path, rotated_volume)

            else:
                # Show a message box if no file is selected
                self.prompt_error("No file selected.")
        
        self.prompt_message("All files have been aligned.")

    