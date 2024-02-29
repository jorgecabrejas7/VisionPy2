# Import necessary modules
from base_plugin import BasePlugin
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from utils import image_sequence
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.measure import regionprops
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

class Plugin(BasePlugin):
    def run(self):
        """
        Prompt the user to select a file and process the selected file.

        Raises:
            FileNotFoundError: If no file is selected.

        Example:
            run()
        """

        # Open a file dialog to select a file
        file_path = self.select_folder(
            "Select Folder for volume 1"
        )
        # Open a file dialog to select a file
        file_path2 = self.select_folder(
            "Select Folder for volume 2"
        )

        # Open a file dialog to select a folder to save the concatenated volume
        save_path = self.select_folder(
            "Select Folder to save concatenated volume"
        )

        # Check if a file was selected
        if file_path and file_path2:

            print('the path 1 is: ', file_path)
            print('the path 2 is: ', file_path2)

            # Create a progress window

            self.update_progress(0, "Loading Volume 1")

            volume1 = image_sequence.read_sequence2(file_path,progress_window=self)

            self.update_progress(100, "Loading Volume 1",1,1)

            # Create a progress window

            self.update_progress(0, "Loading Volume 2")

            volume2 = image_sequence.read_sequence2(file_path2,progress_window=self)

            self.update_progress(100, "Loading Volume 2",1,1)

            def flip_ask(volume2):

                #Create a window that asks the second volume to be flipped
                msg_box = QMessageBox(self.main_window)
                msg_box.setText("Do you want to flip the second volume?")
                msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)
                ret = msg_box.exec()
                #if selected yes, flip the volume
                if ret == QMessageBox.StandardButton.Yes:
                    return volume2[::-1]
                else:
                    return volume2
            
            volume2 = self.request_gui(flip_ask, volume2)

            i,j =self.request_gui(self.get_candidates,volume1.shape[0],volume2.shape[0],show=False)

            # Select thresholds
            user_inputs = self.request_gui(self.get_thresholds,volume1,volume2,i,j)

            if len(user_inputs) > 0:

                #main
                # Create a progress window

                self.update_progress(0, "Shifting Volume")
                #get shift
                shift = user_inputs['Main']
                #shift volume
                volume2 = self.shift_volume_concurrent(volume2,shift,progress_window=self)
                
                concatenated = self.concatenate_volumes(volume1,volume2,i,j)

                # Check if a folder was selected
                if save_path:

                    # Create a progress window
                    # Save the centered volume
                    self.update_progress(0, "Saving concatenated Volume")
                    image_sequence.write_sequence2(save_path, os.path.basename(save_path), concatenated,progress_window=self)


                    # Show a message box if the centered volume is saved successfully, when the message box is closed, close all matplotlib figures
                    self.prompt_message("Concatenated volume saved successfully.")

                    self.update_progress(100, "Saving concatenated Volume",1,1)



                
    
    def get_candidates(self,vol1shape,vol2shape,show=False):
        #ask the user for length(int), resolution (decimal) and frameid(int)
        length, ok = QInputDialog.getInt(self.main_window, "Length", "Enter the length of the sample in millimeters:")
        resolution, ok = QInputDialog.getDouble(self.main_window, "Resolution", "Enter the resolution of the scan in millimeters:",decimals = 5)

        #calculate overlapping region
        lengtha = resolution * vol1shape
        end_top = length - lengtha
        slice1 = int(end_top / resolution)

        #show a message box that shows the overlapping region
        msg_box = QMessageBox(self.main_window)
        msg_box.setText(f"Overlap in volume1 since slice {slice1}")
        msg_box.exec()

        while True:

            frameid, ok = QInputDialog.getInt(self.main_window, "Frameid", "Frameid of slice to search candidates \nFrom " + str(slice1) + " to " + str(vol1shape-1) + ":")

            if frameid >= slice1 and frameid < vol1shape:
                break

            self.prompt_error("Invalid slice id")
        

        frameid = np.clip(frameid,a_min=slice1,a_max=vol1shape-1)

        #find candidates
        candidates = self.find_candidates_slice(frameid,length,resolution,vol2shape,show=True)

        candidates = candidates[:,1]

        #show the first and last candidate to the user in a message box
        msg_box = QMessageBox(self.main_window)
        msg_box.setText(f"First candidate: {candidates[0]}\nLast candidate: {candidates[-1]}")
        msg_box.exec()

        while True:

            frameid, ok = QInputDialog.getInt(self.main_window, "Frameid", "Enter the frame of the first volume to concatenate:")
            frameid2, ok = QInputDialog.getInt(self.main_window, "Frameid2", "Enter the frame of the second volume to concatenate:")

            if frameid >= slice1 and frameid < vol1shape and frameid2 >= 0 and frameid2 < candidates[-1]:
                break
            
            self.prompt_error("Invalid slice id")


        return frameid,frameid2
        
    
    def find_candidates_slice(self,frameid,length,resolution,vol2shape,range_slices=200,show=False):
        """
        Find the candidates for a given frameid, length and resolution.

        Args:
        frameid (int): Frame id of the first volume.
        length (int): Length of the sample scanned in two volumes in millimeters.
        resolution (int): Resolution of the scan.
        vol1shape (int): Number of slices in the first volume.
        vol2shape (int): Number of slices in the second volume.

        Returns:
        list: List of candidates to be the same slice in the second volume as frameid in the first.
        """

        #Calculate the number of slices needed to cover the length of the sample
        n_slices = int(length/resolution)

        #Calulate the remaining slices needed to cover the length of the sample 
        remaining = n_slices - frameid

        #Calculate where should the first slice of he second volume be to fit the length of the sample
        start_vol2 = vol2shape - remaining

        #print all
        if show:
            print(f"frameid: {frameid}, n_slices: {n_slices}, remaining: {remaining}, start_vol2: {start_vol2}")

        #define range
        start = np.clip(start_vol2-range_slices//2,a_min=0,a_max=None)
        end = np.clip(start_vol2+range_slices//2,a_min=None,a_max=vol2shape)


        #Create a list of candidates, all numbers in a range of 200 being start_vol2 the center
        candidates = [i for i in range(start,end)]

        #Create tuples of one candidate and the frameid
        return np.array([(frameid,candidate) for candidate in candidates])
    
    def get_user_inputs(self, name, default_value=65535):
        #Get the threshold value for the name reslice
        threshold, ok = QInputDialog.getInt(self.main_window, "Threshold", f"Enter Threshold for {name} reslice:", default_value)

        if ok:
            return threshold
        else:
            return None

    def get_thresholds(self,volume1,volume2,i,j):

        #detect if the volume is 8bits or 16bits
        if volume1.dtype == np.uint8:
            is8bit = True
        else:
            is8bit = False

        names = ['Main']

        user_inputs = {'Main':None}

        if is8bit:
            top_threshold = 255
        else:
            top_threshold = 65535

        for name in names:

            while True:

                top_threshold = self.get_user_inputs(name, default_value=top_threshold)

                #Check if the threshold is valid by aplying it to the middle slice
                if top_threshold != None:

                    s1 = volume1[i]
                    s2 = volume2[j]

                    #copy slices
                    slice1 = s1.copy()
                    slice2 = s2.copy()
                    
                    #make 0 every pixel above the threshold in each slice
                    slice1[slice1>top_threshold] = 0
                    slice2[slice2>top_threshold] = 0

                    #autothreshold the slices using otsu
                    thresh1 = threshold_otsu(slice1)
                    thresh2 = threshold_otsu(slice2)

                    #apply threshold to the slices
                    slice1 = slice1 > thresh1
                    slice2 = slice2 > thresh2

                    #label the slices  
                    slice1 = label(slice1)
                    slice2 = label(slice2)

                    #get the largest component of each slice
                    regprops1 = regionprops(slice1)
                    regprops2 = regionprops(slice2)

                    if (regprops1 == []) or (regprops2 == []):
                        #Show a message box if invalid threshold is selected
                        msg_box = QMessageBox(self.main_window)
                        msg_box.setText("To low threshold selected.")
                        msg_box.exec()
                        continue

                    largest1 = regprops1[np.argmax([regprop.area for regprop in regprops1])].label
                    largest2 = regprops2[np.argmax([regprop.area for regprop in regprops2])].label
                    slice1[slice1!=largest1] = 0
                    slice2[slice2!=largest2] = 0

                    #calculate the cross correlation between the two slices
                    corr = np.fft.fftshift(np.fft.fft2(slice1))*np.fft.fftshift(np.fft.fft2(slice2).conj())

                    #calculate the inverse fourier transform of the cross correlation
                    corr = np.fft.ifft2(corr)

                    #calculate the shift in x and y
                    shift = np.unravel_index(np.argmax(corr), corr.shape)

                    print(f"shift: {shift}, corr.shape: {corr.shape}")

                    #shift the second slice so it matches the first
                    shifted = np.roll(slice2, shift[0], axis=0)
                    shifted = np.roll(shifted, shift[1], axis=1)

                    #plot the two slices using matplotlib
                    fig, ax = plt.subplots(1,2,figsize=(10,10))
                    ax[0].imshow(slice1, cmap='gray')
                    ax[1].imshow(shifted, cmap='gray')
                    plt.show()

                    #Ask the user if the threshold is ok
                    msg_box = QMessageBox(self.main_window)
                    msg_box.setText(f"Is the threshold ok for {name} reslice?")
                    msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)
                    ret = msg_box.exec()

                    #close the figure
                    plt.close()

                    if ret == QMessageBox.StandardButton.Yes:
                        user_inputs[name] = shift
                        break
                    else:
                        continue
                else:
                    #Show a message box if no threshold is selected
                    msg_box = QMessageBox(self.main_window)
                    msg_box.setText("No threshold selected.")
                    msg_box.exec()
        
        return user_inputs

    def shift_volume_concurrent(self,volume,shift, progress_window=None):
        """
        Apply autothreshold_slice to each slice in a volume concurrently with a progress bar.
        
        Args:
        volume (numpy.ndarray): A 3D array where each slice corresponds to an image.
        
        Returns:
        numpy.ndarray: A 3D array where each slice is shifted.
        """

        def shift_slice(args):
            """
            Shift a slice.

            Args:
            slice (numpy.ndarray): Slice to be shifted.
            shift (tuple): Tuple with the x and y shift.

            Returns:
            numpy.ndarray: Shifted slice.
            """
            #unpack args
            slice, shift = args
            #shift the slice
            shifted = np.roll(slice, shift[0], axis=0)
            shifted = np.roll(shifted, shift[1], axis=1)

            #return the shifted slice
            return shifted

        shifted_volume = np.zeros_like(volume)

        if progress_window == None:
            with ThreadPoolExecutor() as executor:
                args = [(volume[i], shift) for i in range(volume.shape[0])]
                for i, result in enumerate(executor.map(shift_slice, args)):
                    shifted_volume[i] = result
        else: 
            with ThreadPoolExecutor() as executor:
                args = [(volume[i], shift) for i in range(volume.shape[0])]
                for i, result in enumerate(executor.map(shift_slice, args)):
                    shifted_volume[i] = result
                    progress_window.update_progress(int(i / volume.shape[0] * 100), f"Shifting: {i}",i,volume.shape[0])
        
        return np.array(shifted_volume)
    
    def concatenate_volumes(self,vol1,vol2,i,j):
        """
        Concatenate two volumes.

        Args:
        vol1 (numpy.ndarray): First volume.
        vol2 (numpy.ndarray): Second volume.
        i (int): Slice id of the first volume.
        j (int): Slice id of the second volume.

        Returns:
        numpy.ndarray: Concatenated volume.
        """
        #concatenate the two volumes
        concatenated = np.concatenate((vol1[:i],vol2[j:]),axis=0)

        #return the concatenated volume
        return concatenated