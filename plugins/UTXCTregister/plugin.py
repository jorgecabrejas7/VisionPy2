# Import necessary modules
from base_plugin import BasePlugin
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from utils import image_sequence
import matplotlib.pyplot as plt
import tifffile as tiff
import plugins.UTXCTregister.UTXCTregister as reg
import numpy as np
import time


# Define a class that implements the PluginInterface
class Plugin(BasePlugin):
    # Prompt the user to select a file
    def run(self):

        batchmode = True

        files_ut_paths = []
        files_xct_paths = []
        saves_paths = []

        while batchmode:

            # Open a file dialog to select a file
            files_ut_paths.append(self.select_file("Select UT File"))

            # Open a file dialog to select a file

            files_xct_paths.append(self.select_file("Select XCT File"))

            #open a file dialog to select a save path
            saves_paths.append(self.select_folder("Select a folder to save the files."))

            batchmode = self.prompt_confirmation("Do you want to use more files?")
        
        for ut_path,xct_path,save_path in zip(files_ut_paths,files_xct_paths,saves_paths):

            # Check if a file was selected
            if ut_path and xct_path and save_path:

                #load the files
                ut = tiff.imread(ut_path)
                ut = np.swapaxes(ut, 0, 1)
                ut = np.swapaxes(ut, 1, 2)
                xct = tiff.imread(xct_path)
                xct = np.swapaxes(xct, 0, 1)
                xct = np.swapaxes(xct, 1, 2)

                print('Loaded')

                #preprocess the files
                ut_centers = reg.ut_preprocessing(ut)
                xct_centers = reg.xct_preprocessing(xct)

                print('preprocessed')

                #label the images
                ut_labeled = reg.label_objects(ut_centers)
                xct_labeled = reg.label_objects(xct_centers)

                print('labeled')

                #print the actual time, the hour and minute of the day
                print(time.strftime("%H:%M:%S"))

                #register
                transformation = reg.register(ut_labeled, xct_labeled, save_path)

                print('registered')

                #apply the transformation
                xct_aligned = reg.apply_transform(transformation, ut, xct)

                print('applied registration')
                
                image_sequence.write_sequence2(save_path,'registered_xct', xct_aligned)

            else:
                # Show a message box if no file is selected
                self.prompt_error("No file selected.")
        
        self.prompt_message("All files have been registered.")

    