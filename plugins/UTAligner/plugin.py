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
import tifffile as tiff
# Define a class that implements the PluginInterface
class Plugin(BasePlugin):
    # Prompt the user to select a file
    def run(self):

        # Open a file dialog to select a file
        file_path = self.select_file(
            "Select UT File"
        )

        # Open a file dialog to select a folder to save the rotated volume
        save_path = self.select_save_file(
            "Select filename to save the aligned volume"
        )

        # Check if a file was selected
        if file_path:

            print('the path is: ', file_path)

            # Create a progress window

            self.update_progress(0, "Loading Volume")

            volume = tiff.imread(file_path)

            gate = self.get_gate(volume)

            self.update_progress(100, "Loading Volume",1,1)

            volume = self.align(volume,gate)

            self.update_progress(100, "Aligning Volume",1,1)

            if save_path:
                    
                # Create a progress window

                # Save the rotated volume
                self.update_progress(0, "Saving Aligned Volume")

                tiff.imsave(save_path, volume)

                # Show a message box if the rotated volume is saved successfully, when the message box is closed, close all matplotlib figures
                self.update_progress(100, "Saving Aligned Volume",1,1)
                self.prompt_message("Aligned volume saved successfully.")
                self.request_gui(plt.close,'all')


    def ask_gate(self):

        #create a window 
        start, ok = QInputDialog.getInt(self.main_window, "Start", f"Enter start of the gate",0)
        end, ok = QInputDialog.getInt(self.main_window, "End", f"Enter start of the gate",0)

        if ok:
            return (start,end)
    
    def plot_signal_gate(self,signal,gate):
        #clean the plot
        plt.close('all')
        plt.plot(signal)
        plt.axvline(x=gate[0], color='r', linestyle='--')
        plt.axvline(x=gate[1], color='r', linestyle='--')
        plt.show()

    def get_gate(self, data):

        max_signal = np.max(data, axis=(1,2))

        self.request_gui(plt.plot,max_signal)

        while True:

            while True:

                gate = self.request_gui(self.ask_gate)

                #gate end must be greater than gate start and start must be greater than 0
                if (gate[0] < gate[1]) and (gate[0] > 0):
                    break
            
            #plot the signal and the gate

            self.request_gui(self.plot_signal_gate,max_signal,gate)

            #ask the user if the gate is correct
            response = self.prompt_confirmation("Is the gate correct?")

            print(response)

            if response:
                return gate



            
    
    
    def align(self,data,gate):
        #now do it for the whole volume
        rolled_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                signal = data[:,i,j]
                gated_data = signal[gate[0]:gate[1]]
                max_gated_data_index = np.argmax(gated_data, axis=0)
                rolled = np.roll(signal, -max_gated_data_index)
                rolled_data[:,i,j] = rolled
        return rolled_data