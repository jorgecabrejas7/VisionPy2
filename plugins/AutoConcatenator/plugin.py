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
import scipy.ndimage as ndimage
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor
#load vgg16 model and remove classification layers
from tensorflow.keras.applications.mobilenet import MobileNet
from scipy.spatial import distance as dist
from skimage.transform import resize
from pystackreg import StackReg

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
            
            def registered_ask():

                #creat a window that asks the user if the volumes are registered
                msg_box = QMessageBox(self.main_window)
                msg_box.setText("Are the volumes registered?")
                msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)
                ret = msg_box.exec()
                #if selected yes, return False
                if ret == QMessageBox.StandardButton.Yes:
                    return False
                else:
                    return True
            
            self.request_gui(flip_ask, volume2)

            register = self.request_gui(registered_ask)

            length, resolution, start, end, search_range = self.request_gui(self.get_info, volume1.shape[0],volume2.shape[0],show=False)

            print(f"length: {length}, resolution: {resolution}, start: {start}, end: {end}")

            rois = self.get_volume_bbox(volume1)

            distancias_concurrente = self.compare_slices_concurrent_ai(volume1,volume2,length,resolution,start =start, end =end, rois=rois, n_chunks = 1, range_slices=search_range)

            #get the minimum distance of distancias_concurrente
            min_distance = np.argmin(distancias_concurrente[:,2])
            i = int(distancias_concurrente[min_distance][0])
            j = int(distancias_concurrente[min_distance][1])
            distance = distancias_concurrente[min_distance][2]
            #print all
            print(f"i: {i}, j: {j}, distance: {distance}")

            self.prompt_message(f"i: {i}, j: {j}, distance: {distance}")

            if len([1]) > 0:

                if register:

                    # Create a progress window
                    middle = volume1[i]

                    volume2 = self.stackreg(volume2[j:],middle,progress_window=self)

                    self.update_progress(100, "Registering Volume",1,1)

                concatenated = self.concatenate_volumes(volume1,volume2,i,j)

                # Check if a folder was selected
                if save_path:

                    # Create a progress window

                    # Save the centered volume
                    self.update_progress(0, "Saving concatenated Volume")
                    image_sequence.write_sequence2(save_path, os.path.basename(save_path), concatenated,progress_window=self)

                    # Show a message box if the centered volume is saved successfully, when the message box is closed, close all matplotlib figures
                    self.update_progress(100, "Saving concatenated Volume",1,1)
                    self.prompt_message("Concatenated volume saved successfully.")
                    



                
    
    def get_info(self,vol1shape,vol2shape,show=False):
        #ask the user for length(int), resolution (decimal) and frameid(int)
        length, ok = QInputDialog.getInt(self.main_window, "Length", "Enter the length of the sample in millimeters:")
        resolution, ok = QInputDialog.getDouble(self.main_window, "Resolution", "Enter the resolution of the scan in millimeters:",decimals = 5)
        search_range, ok = QInputDialog.getInt(self.main_window, "Search Range", "Enter the slices search range:")
        #calculate overlapping region
        lengtha = resolution * vol1shape

        lengthb = resolution * vol2shape

        end_top = length - lengtha

        slice1 = int(end_top / resolution)

        return [length, resolution, slice1,vol1shape,search_range]
        
    
    def find_candidates_slice(self,frameid,length,resolution,vol1shape,vol2shape,range_slices=200,show=False):
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

    def register_slices(self,args):
        """
        Register two slices using the OpenCV phase correlation method.
        
        Args:
        slice1 (numpy.ndarray): A 2D array corresponding to a slice.
        slice2 (numpy.ndarray): A 2D array corresponding to a slice.
        
        Returns:
        numpy.ndarray: A 2D array corresponding to the registered slice.
        """
        slice1, slice2, sr = args
        t = slice1.dtype
        out = sr.transform(slice2).astype(t)
        
        return out
    
    def stackreg(self,volume,reference, progress_window=None):

        # Create a StackReg object
        sr = StackReg(StackReg.TRANSLATION)
        middle = volume[0]
        a = sr.register_transform(reference, middle)

        #Create a list of indexes in pairs
        indexes = [(i) for i in range(volume.shape[0])]

        volume_reg = np.zeros_like(volume)

        if progress_window == None:
            for idx in indexes:
                #Register the slices
                slice2_reg = self.register_slices((reference, volume[idx],sr))
                #Add the registered slice to the registered volume
                volume_reg[idx] = slice2_reg
        else:
            for idx in indexes:
                #Register the slices
                slice2_reg = self.register_slices((reference, volume[idx],sr))
                #Add the registered slice to the registered volume
                volume_reg[idx] = slice2_reg
                progress_window.update_progress(int(idx / len(indexes) * 100), f"Registering: {idx}",idx,volume.shape[0])

        return volume_reg

    def compare_slices_concurrent_ai(self,volume1, volume2,length,resolution,rois,start=0,end=-1, n_chunks = 100,range_slices=200):
        """
        Compare every slice of two volumes concurrently and return a list of distances.

        Args:
        volume1 (numpy.ndarray): First volume.
        volume2 (numpy.ndarray): Second volume.

        Returns:
        list: List of distances between each pair of slices.
        """

        def prepare_images_for_model(img1):

            #resize images to 224x224
            img1 = resize(img1, (224,224))

            #prepare the images for the model, they are greyscale so we add a channel dimension

            img1 = np.expand_dims(img1, axis=-1)
            img1 = np.repeat(img1, 3, axis=-1)

            #prepare them for the model

            img1 = img1.reshape((1,224,224,3))

            return img1
        
        def compare_features(args):

            features1,features2,i,j = args

            d1 = dist.euclidean(features1.flatten(), features2.flatten())

            return i,j,d1


        #initialize model

        #create a progress window
        self.update_progress(0, "Loading Model")

        # load model without classifier layers for greyscale images
        model = MobileNet(include_top=False, input_shape=(224, 224, 3))
        
        distances = []

        self.update_progress(0, "Looking for candidates")

        #List indexes of volume 1 only
        indexes = [(i+start) for i in range(volume1[start:end].shape[0])]

        #print(indexes)

        #find candidates
        indexes = self.find_candidates_volume(volume1,volume2,length,resolution,indexes,range=range_slices)

        #get the first and last candidate
        first = indexes[0][1]
        last = indexes[-1][1]

        print(first,last)
        
        images1 = [volume1[i] for i in range(start,end)]
        images2 = [volume2[i] for i in range(first,last)]

        self.update_progress(0, "Preparing images for model")

        #prepare images for model concurrently
        with ThreadPoolExecutor() as executor:
            images1_prepared = list(executor.map(prepare_images_for_model, images1))
            images2_prepared = list(executor.map(prepare_images_for_model, images2))

        self.update_progress(0, "Getting features from images")

        #get features from images
        features1 = model.predict(np.array(images1_prepared).reshape((-1,224,224,3)))
        features2 = model.predict(np.array(images2_prepared).reshape((-1,224,224,3)))

        self.update_progress(0, "Preparing arguments")

        #prepare args for compare_features
        args = [(features1[i-start-1],features2[j-first-1],i,j) for i,j in indexes]

        # Create a thread pool executor
        with ThreadPoolExecutor() as executor:
            # Create a list of arguments for each pair of slices with a progress bar
            for distance in executor.map(compare_features, args):
                distances.append(np.array(distance))
                i = distance[0]
                self.update_progress(int((i-start) / (end-start) * 100), f"Comparing: {i-start}",i-start,end-start)
                
        plt.close()
        
        return np.array(distances)
    
    def compare_slices_concurrent_2(self,volume1, volume2,length,resolution,rois,start=0,end=-1, n_chunks = 100,range_slices=200, progress_window=None):
        """
        Compare every slice of two volumes concurrently and return a list of distances.

        Args:
        volume1 (numpy.ndarray): First volume.
        volume2 (numpy.ndarray): Second volume.

        Returns:
        list: List of distances between each pair of slices.
        """

        #define threadable functions
        def compare_images_all(args,show=False):

            def compare_images(args):
                """
                Compare two images using the chi-square metric.
                
                Args:
                image1 (numpy.ndarray): First image.
                image2 (numpy.ndarray): Second image.
                
                Returns:
                float: Chi-square distance between the two images.
                """
                image1, image2, i , j  = args

                # Flatten the images
                image1_flat = image1.flatten()
                image2_flat = image2.flatten()
                
                # Calculate the histogram of each image
                hist1, _ = np.histogram(image1_flat, bins=256, range=(0, 256))
                hist2, _ = np.histogram(image2_flat, bins=256, range=(0, 256))

                #normalize histograms
                hist1 = hist1 / np.sum(hist1)
                hist2 = hist2 / np.sum(hist2)
                
                # Calculate the chi-square distance
                chi_square = np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-10))
                
                return i,j,chi_square

            def compare_images_3(args):
                """
                Compare two images using the chi-square metric.
                
                Args:
                image1 (numpy.ndarray): First image.
                image2 (numpy.ndarray): Second image.
                
                Returns:
                float: Chi-square distance between the two images.
                """
                image1, image2, i , j  = args

                # Pad the smaller image with zeros to match the size of the larger image
                if image1.shape != image2.shape:
                    max_height = max(image1.shape[0], image2.shape[0])
                    max_width = max(image1.shape[1], image2.shape[1])
                    image1_padded = np.zeros((max_height, max_width))
                    image2_padded = np.zeros((max_height, max_width))
                    image1_padded[:image1.shape[0], :image1.shape[1]] = image1
                    image2_padded[:image2.shape[0], :image2.shape[1]] = image2
                    image1 = image1_padded
                    image2 = image2_padded

                #convert to 8bit
                image1 = image1.astype(np.uint8)
                image2 = image2.astype(np.uint8)

                similarity = ssim(image1, image2, multichannel=False)

                return i,j,-similarity


            image1, image2, i , j, rois = args
            image1[image1 < 120] = 0
            image2[image2 < 120] = 0
            distance = 0
            for r in rois:
                y0,x0,y1,x1 = r
                args1 =[image1[min(x0,x1):max(x0,x1),min(y0,y1):max(y0,y1)],image2[min(x0,x1):max(x0,x1),min(y0,y1):max(y0,y1)],i,j]
                distance_a = np.array([compare_images(args1), compare_images_3(args1)])[:,-1].sum()
                #show using pyplot the two images of args1
                if show:
                    plt.subplot(121)
                    plt.imshow(args1[0], cmap='gray')
                    plt.title(str(distance_a))
                    plt.subplot(122)
                    plt.imshow(args1[1], cmap='gray')
                    plt.title('image2')
                    plt.show()
                distance+=distance_a
            return i,j,distance
        
        # Create a list of distances
        distances = []

        #List indexes of volume 1 only
        indexes = [(i+start) for i in range(volume1[start:end].shape[0])]

        #print(indexes)

        #find candidates

        indexes = self.find_candidates_volume(volume1,volume2,length,resolution,indexes,range=range_slices)

        #divide the indexes in chunks
        chunks = np.array_split(indexes, n_chunks)

        print(end-start)

        for indexes in chunks:

            distances_aux = []

            if progress_window == None:
                # Create a thread pool executor
                with ThreadPoolExecutor() as executor:
                    # Create a list of arguments for each pair of slices with a progress bar
                    args = [(volume1[i], volume2[j],i,j,rois) for i, j in indexes]
                    # Compare each pair of slices concurrently
                    for distance in executor.map(compare_images_all, args):
                        distances_aux.append(np.array(distance))
            else:
                # Create a thread pool executor
                with ThreadPoolExecutor() as executor:
                    # Create a list of arguments for each pair of slices with a progress bar
                    args = [(volume1[i], volume2[j],i,j,rois) for i, j in indexes]
                    # Compare each pair of slices concurrently
                    for distance in executor.map(compare_images_all, args):
                        distances_aux.append(np.array(distance))
                        i = distance[0]
                        progress_window.update_progress(int((i-start) / (end-start) * 100), f"Comparing: {i-start}",i-start,end-start)
            
            #get the minimum distance of distances_aux
            distances_aux = np.array(distances_aux)
            min_distance = np.argmin(distances_aux[:,2])
            distances.append(distances_aux[min_distance])

        return np.array(distances)

    def find_candidates_volume(self,vol1,vol2,length,resolution,indexes,range=200):

        candidates = []

        for i in indexes:
            candidates.append(self.find_candidates_slice(i,length,resolution,vol1.shape[0],vol2.shape[0],range_slices=range))
        
        #concatenate all candidates
        candidates = np.concatenate(candidates)

        return candidates