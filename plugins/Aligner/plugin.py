# Import necessary modules
from base_plugin import BasePlugin
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from utils import image_sequence
from pathlib import Path
import traceback
import sys
import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.measure import regionprops
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import binary_fill_holes
from skimage import feature

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
            
            #close the progress window
            self.update_progress(100, "Loading Volume",1,1)

            reslices = self.request_gui(self.get_user_inputs3)

            # Select thresholds
            angles = self.request_gui(self.get_angles, volume=volume, reslices=reslices)

            logging.info('-----------------------')

            # Perform operations on the selected file based on user inputs
            if angles:

                if reslices['Main']:

                    # main
                    # Create a progress window

                    self.update_progress(0, "Rotating Volume from Main reslice")
                    print('The angles: ', angles)
                    angle = angles["Main"]
                    volume = self.rotate_volume_concurrent(volume, angle,progress_window=self)
                    self.update_progress(100, "Rotating Volume from Main reslice")
                
                if reslices['Left']:

                    # Create a progress window
                    self.update_progress(0, "Rotating Volume from Left reslice")
                    volume = np.transpose(volume, (2, 0, 1))
                    angle = angles["Left"]
                    volume = self.rotate_volume_concurrent(volume, angle,progress_window=self)
                    self.update_progress(100, "Rotating Volume from Left reslice")

                    if not reslices['Top']:
                        volume = np.transpose(volume, (2, 0, 1))
                        volume = np.transpose(volume, (2, 0, 1))
                
                if reslices['Top']:

                    # Create a progress window
                    self.update_progress(0, "Rotating Volume from Top reslice")
                    if not reslices['Left']:
                        volume = np.transpose(volume, (2, 0, 1))
                    volume = np.transpose(volume, (2, 0, 1))
                    angle = angles["Top"]
                    volume = self.rotate_volume_concurrent(volume, angle,progress_window=self)
                    self.update_progress(100, "Rotating Volume from Top reslice")

                    volume = np.transpose(volume, (2, 0, 1))

                # Check if a folder was selected
                if save_path:

                    # Create a progress window
                    # Save the rotated volume
                    self.update_progress(0, "Saving Rotated Volume")
                    image_sequence.write_sequence2(save_path, os.path.basename(save_path), volume,progress_window=self)
                    
                    # Show a message box if the rotated volume is saved successfully, when the message box is closed, close all matplotlib figures
                    self.prompt_message("The rotated volume is saved successfully.")
                    self.request_gui(plt.close, "all")
                    self.update_progress(100, "Saving Rotated Volume",1,1)
                
                else:
                    # Show a message box if no folder is selected
                    self.prompt_message("No saving folder is selected.")


            else:
                # Show a message box if user inputs are not provided
                self.prompt_message("User inputs are not provided.")

        else:
            self.prompt_message("No file is selected.")

        return
    
    def get_lines(self,image):

        import cv2

        #convert image type to uint8
        image = image.astype(np.uint8)

        # Perform Hough Line Transform
        lines = cv2.HoughLines(image,1,np.pi/180,200)

        # Draw lines on the image
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
        
        #show image using matplotlib and creating a figure
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        plt.show()
        
    def get_lines2(self,image):

        print(image.shape)

        #get the first and last pixel of the image
        linea = np.where(image >= 1)
        x0 = linea[0][0]
        x1 = linea[0][-1]
        y0 = linea[1][0]
        y1 = linea[1][-1]

        #create a image with the same size as the input image
        image2 = np.zeros_like(image)

        import cv2

        #draw the line on the image2
        cv2.line(image2,(y0,x0),(y1,x1),(1,1,1),2)

        return image2
    
    def get_user_inputs3(self):

        #Create a window with 3 checkboxes and an Ok button
        window = QDialog()
        window.setWindowTitle("Select Reslices")
        window.setWindowModality(Qt.WindowModality.WindowModal)
        window.resize(300, 100)
        window.setLayout(QVBoxLayout())

        main_checkbox = QCheckBox("Main")
        left_checkbox = QCheckBox("Left")
        top_checkbox = QCheckBox("Top")

        ok_button = QPushButton("Ok")

        window.layout().addWidget(main_checkbox)
        window.layout().addWidget(left_checkbox)
        window.layout().addWidget(top_checkbox)
        window.layout().addWidget(ok_button)
        ok_button.clicked.connect(window.close)
        window.exec()

        main = main_checkbox.isChecked()
        left = left_checkbox.isChecked()
        top = top_checkbox.isChecked()

        return {'Main':main,'Left':left,'Top':top}
    
    def get_angles(self,volume,reslices):
        """
        Get the threshold values for different slices of a volume.

        Args:
        - self: the object instance
        - volume: a 3D numpy array representing the volume

        Returns:
        - user_inputs: a dictionary containing the threshold values for different slices

        Raises:
        - No specific exceptions are raised within this function.

        Example:
        ```python
        # Create an instance of the class
        instance = ClassName()

        # Define a 3D numpy array representing the volume
        volume = np.random.rand(10, 10, 10)

        # Call the function to get the threshold values
        thresholds = instance.get_angles(volume)
        ```
        """

        #detect if the volume is 8bits or 16bits
        if volume.dtype == np.uint8:
            is8bit = True
        else:
            is8bit = False
        
        #names of the keys in the dictionary which value is True
        names = [key for key, value in reslices.items() if value == True]

        angles = {'Main':None,'Left':None,'Top':None}

        if not reslices['Main']:
            if not reslices['Left']:
                volume = np.transpose(volume, (2, 0, 1))
                volume = np.transpose(volume, (2, 0, 1))
            else:
                volume = np.transpose(volume, (2, 0, 1))


        if is8bit:
            top_threshold = 255
        else:
            top_threshold = 65535

        for name in names:

            while True:

                if name == 'Top':
                    if not reslices['Left'] and reslices['Main']:
                        volume = np.transpose(volume, (2, 0, 1))
                
                top_threshold = self.get_user_inputs(name, default_value=top_threshold)

                #Check if the threshold is valid by aplying it to the middle slice
                if top_threshold != None:

                    middle_slice = volume[len(volume)//2].copy()

                    if name == 'Top':
                        #rotate the middle slice by 90 degree
                        middle_slice = np.rot90(middle_slice)

                    #crop the middle slice to its half to avoid the background
                    middle_slice = middle_slice[middle_slice.shape[0]//4:middle_slice.shape[0]//4*3,:]

                    top_index = np.where(middle_slice > top_threshold)

                    middle_slice[top_index] = 0

                    if is8bit:
                        threshold_value = threshold_otsu(middle_slice[middle_slice > 10])
                    else:
                        threshold_value = threshold_otsu(middle_slice[middle_slice > 10000])

                    print('threshold value is: ', threshold_value)

                    thresholded_slice = middle_slice > threshold_value

                    thresholded_slice_to_print = thresholded_slice.copy()

                    # Label the objects in the thresholded slice
                    labeled_slice = label(thresholded_slice)

                    # Get the properties of each labeled region
                    regions = regionprops(labeled_slice)

                    if regions == []:
                        #Show a message box if invalid threshold is selected
                        msg_box = QMessageBox(self.main_window)
                        msg_box.setText("To low threshold selected.")
                        msg_box.exec()
                        continue

                    # Find the largest connected component
                    largest_component = max(regions, key=lambda region: region.area)

                    # Create a mask to keep only the largest component
                    mask = np.zeros_like(labeled_slice)
                    mask[labeled_slice == largest_component.label] = 1
                    mask = binary_fill_holes(mask).astype(int)

                    # Apply the mask to the thresholded slice
                    thresholded_slice = thresholded_slice * mask

                    # extract edges using canny edge detector
                    mask = feature.canny(mask>0, sigma=0) > 0

                    # Label the objects in the thresholded slice
                    mask = label(mask)
                    #print np unique of mask
                    print(np.unique(mask))

                    # Get the properties of each labeled region
                    regions = regionprops(mask)

                    # Find the second largest connected component if there is one, if not the first
                    if len(regions) == 1:
                        second_largest_component = regions[0]
                    else:
                        second_largest_component = sorted(regions, key=lambda region: region.area)[-1]
                    
                    #delete everything that is not the second largest component from mask
                    mask[mask != second_largest_component.label] = 0

                    try:
                        mask = self.get_lines2(mask)
                    except:
                        print('line not smoothed')

                    # Compute the rotation angle of the largest component
                    rotation_angle = second_largest_component.orientation

                    # Convert the angle from radians to degrees
                    rotation_angle_degrees = np.degrees(rotation_angle)

                    rotated_slice = rotate(thresholded_slice, -rotation_angle_degrees)

                    # Print the rotation angle
                    print(f"The rotation angle of the largest component is {rotation_angle_degrees} degrees.")

                    plt.figure()
                    # Plotting the original middle slice
                    plt.subplot(2, 2, 1)
                    plt.imshow(middle_slice, cmap='gray')
                    plt.title('Original Middle Slice')

                    # Plotting the thresholded slice
                    plt.subplot(2, 2, 2)
                    plt.imshow(thresholded_slice_to_print, cmap='gray')
                    plt.title('Thresholded Slice')

                    # Plotting the largest component mask
                    plt.subplot(2, 2, 3)
                    plt.imshow(mask, cmap='gray')
                    plt.title('Largest Component Mask')

                    # Plotting the rotated slice
                    plt.subplot(2, 2, 4)
                    plt.imshow(rotated_slice, cmap='gray')
                    plt.title('Rotated Slice')

                    # Adjusting the layout and displaying the plot
                    plt.tight_layout()
                    plt.show()

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
                        angles[name] = -rotation_angle_degrees
                        volume = np.transpose(volume, (2, 0, 1))
                        break
                    else:
                        continue
                else:
                    #Show a message box if no threshold is selected
                    msg_box = QMessageBox(self.main_window)
                    msg_box.setText("No threshold selected.")
                    msg_box.exec()

        return angles

                
                

    def get_user_inputs(self, name, default_value=65535):
        #Get the threshold value for the name reslice
        threshold, ok = QInputDialog.getInt(self.main_window, "Threshold", f"Enter Threshold for {name} reslice:", default_value)

        if ok:
            return threshold
        else:
            return None
    
    def rotate_volume(self,volume, angle, progress_window=None):
        """
        Rotate a 3D volume by a given angle.
        
        Args:
        volume (numpy.ndarray): A 3D array where each slice corresponds to an image.
        angle (float): Angle in degrees.
        
        Returns:
        numpy.ndarray: A rotated 3D array.
        """

        rotated = rotate(volume[0], angle)

        shape = shape = (volume.shape[0],) + rotated.shape

        rotated_volume = np.zeros(shape=shape, dtype=volume.dtype)

        print(shape)

        if progress_window == None:
            for i in range(volume.shape[0]):
                rotated_volume[i] = rotate(volume[i], angle)
        # Rotate each slice of the volume with progress bar
        else:
            for i in range(volume.shape[0]):
                rotated_volume[i] = rotate(volume[i], angle)
                progress_window.update_progress(int(i / volume.shape[0] * 100), f"Rotating: {i}",i,volume.shape[0])
                
        return rotated_volume


    def rotate_volume_concurrent(self,volume, angle, progress_window=None):

        def rotate_slice(args):
            slice, angle = args
            return rotate(slice, angle)

        rotated = rotate(volume[0], angle)
        shape = (volume.shape[0],) + rotated.shape
        rotated_volume = np.zeros(shape=shape, dtype=volume.dtype)

        if progress_window == None:
            with ThreadPoolExecutor() as executor:
                args = [(volume[i], angle) for i in range(volume.shape[0])]
                for i, result in enumerate(executor.map(rotate_slice, args)):
                    rotated_volume[i] = result
        else: 
            with ThreadPoolExecutor() as executor:
                args = [(volume[i], angle) for i in range(volume.shape[0])]
                for i, result in enumerate(executor.map(rotate_slice, args)):
                    rotated_volume[i] = result
                    progress_window.update_progress(int(i / volume.shape[0] * 100), f"Rotating: {i}",i,volume.shape[0])

        return rotated_volume