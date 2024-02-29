# Import necessary modules
from base_plugin import BasePlugin
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from utils import image_sequence
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from pystackreg import StackReg
# Define a class that implements the PluginInterface
class Plugin(BasePlugin):
    # Prompt the user to select a file
    def run(self):

        # Open a file dialog to select a file
        file_path = self.select_folder(
            "Select Folder"
        )

        # Open a file dialog to select a folder to save the rotated volume
        save_path = self.select_folder(
            "Select Folder to save rotated volume"
        )

        # Check if a file was selected
        if file_path:

            print('the path is: ', file_path)

            # Create a progress window

            self.update_progress(0, "Loading Volume")

            volume = image_sequence.read_sequence2(file_path,progress_window=self)

            self.update_progress(100, "Loading Volume",1,1)

            # Create a progress window

            volume = self.stackreg(volume, progress_window=self)

            self.update_progress(100, "Registering Volume",1,1)

            if save_path:
                    
                # Create a progress window

                # Save the rotated volume
                self.update_progress(0, "Saving Registered Volume")
                image_sequence.write_sequence2(save_path, os.path.basename(save_path), volume,progress_window=self)

                
                

                # Show a message box if the rotated volume is saved successfully, when the message box is closed, close all matplotlib figures
                self.update_progress(100, "Saving Registered Volume",1,1)
                self.prompt_message("Registered volume saved successfully.")
                self.request_gui(plt.close,'all')



    
    
    #Create a functiont that given two slices, registers them and returns the registered slice
    def register_slices(self,args):
        """
        Register two slices using the OpenCV phase correlation method.
        
        Args:
        slice1 (numpy.ndarray): A 2D array corresponding to a slice.
        slice2 (numpy.ndarray): A 2D array corresponding to a slice.
        
        Returns:
        numpy.ndarray: A 2D array corresponding to the registered slice.
        """
        slice1, slice2 = args
        t = slice1.dtype
        sr = StackReg(StackReg.TRANSLATION)
        out = sr.register_transform(slice1, slice2).astype(t)
        
        return out

    def stackreg(self,volume, progress_window=None):

        #get middle slice index
        middle = volume.shape[0]//2

        #Create a list of indexes in pairs
        indexes = [(middle, i+1) for i in range(volume.shape[0]-1)]

        volume_reg = np.zeros_like(volume)

        #firslice is the same in both volumes
        volume_reg[0] = volume[0]

        if progress_window == None:
            for idx in indexes:
                #Register the slices
                slice2_reg = self.register_slices((volume[idx[0]], volume[idx[1]]))
                #Add the registered slice to the registered volume
                volume_reg[idx[1]] = slice2_reg
        else:
            for idx in indexes:
                #Register the slices
                slice2_reg = self.register_slices((volume[idx[0]], volume[idx[1]]))
                #Add the registered slice to the registered volume
                volume_reg[idx[1]] = slice2_reg
                progress_window.update_progress(int(idx[1] / len(indexes) * 100), f"Registering: {idx[1]}",idx[1],volume.shape[0])

        return volume_reg