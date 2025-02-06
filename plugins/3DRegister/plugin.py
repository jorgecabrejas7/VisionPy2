# Import necessary modules
from base_plugin import BasePlugin
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from utils import image_sequence
import os
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.measure import regionprops
from scipy.ndimage import binary_fill_holes
import SimpleITK as sitk
import time


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
        file_path = self.select_folder("Select Folder for volume 1")
        # Open a file dialog to select a file
        file_path2 = self.select_folder("Select Folder for volume 2")

        # Open a file dialog to select a folder to save the concatenated volume
        save_path = self.select_folder("Select Folder to save registered volume")

        # Check if a file was selected
        if file_path and file_path2:
            print("the path 1 is: ", file_path)
            print("the path 2 is: ", file_path2)

            self.update_progress(0, "Loading Volume 1")

            bot_full = image_sequence.read_sequence2(file_path, progress_window=self)

            self.update_progress(100, "Loading Volume 1", 1, 1)

            # Create a progress window
            self.update_progress(0, "Loading Volume 2")

            top_full = image_sequence.read_sequence2(file_path2, progress_window=self)

            self.update_progress(100, "Loading Volume 2", 1, 1)

            def flip_ask(volume2):
                # Create a window that asks the second volume to be flipped
                msg_box = QMessageBox(self.main_window)
                msg_box.setText("Do you want to flip the second volume?")
                msg_box.setStandardButtons(
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)
                ret = msg_box.exec()
                # if selected yes, flip the volume
                if ret == QMessageBox.StandardButton.Yes:
                    return volume2[::-1]
                else:
                    return volume2

            top_full = self.request_gui(flip_ask, top_full)

            roi = self.get_roi(bot_full)  # x0,x1,y0,y1

            print(f"Roi: {roi}")

            bot_roi = bot_full[:, roi[0] : roi[1], roi[2] : roi[3]]
            top_roi = top_full[:, roi[0] : roi[1], roi[2] : roi[3]]

            start, end = self.request_gui(
                self.get_info, bot_roi.shape[0], top_roi.shape[0], show=False
            )

            # get the overlapping regions only
            bot_roi_overlapping = bot_roi[start:]
            top_roi_overlapping = top_roi[:end]
            bot_full_overlapping = bot_full[start:]
            top_full_overlapping = top_full[:end]

            print(bot_roi.shape)

            print(f"start: {start}, end: {end}")

            if len([1]) > 0:
                print("Creating masks")
                start = time.time()

                # get the masks
                mask_bot = self.mask(bot_roi_overlapping)
                mask_top = self.mask(top_roi_overlapping)
                mask_bot_full = self.mask(bot_full_overlapping)

                print(f"Time to create masks: {time.time() - start}")

                print("Converting to SimpleITK type")
                start = time.time()

                # Load the fixed and moving images
                fixed_image = sitk.GetImageFromArray(bot_roi_overlapping)
                moving_image = sitk.GetImageFromArray(top_roi_overlapping)
                fixed_image_full = sitk.GetImageFromArray(bot_full_overlapping)
                moving_image_full = sitk.GetImageFromArray(top_full_overlapping)
                final_bot = sitk.GetImageFromArray(bot_full)
                final_top = sitk.GetImageFromArray(top_full)
                fixed_mask_image = sitk.GetImageFromArray(mask_bot)
                moving_mask_image = sitk.GetImageFromArray(mask_top)
                fixed_mask_image_full = sitk.GetImageFromArray(mask_bot_full)

                print(f"Time to convert to SimpleITK type: {time.time() - start}")

                # Initial alignment of the two volumes
                initial_transform = sitk.CenteredTransformInitializer(
                    fixed_image,
                    moving_image,
                    sitk.AffineTransform(3),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY,
                )

                # Set up the registration framework
                registration_method = sitk.ImageRegistrationMethod()

                # mask settings
                registration_method.SetMetricFixedMask(fixed_mask_image)
                registration_method.SetMetricMovingMask(moving_mask_image)

                # Similarity metric settings
                registration_method.SetMetricAsMattesMutualInformation(
                    numberOfHistogramBins=50
                )
                # registration_method.SetMetricAsMeanSquares()
                registration_method.SetMetricSamplingStrategy(
                    registration_method.RANDOM
                )
                registration_method.SetMetricSamplingPercentage(1)

                # Interpolator
                registration_method.SetInterpolator(sitk.sitkLinear)

                # Optimizer settings
                registration_method.SetOptimizerAsRegularStepGradientDescent(
                    learningRate=2.0,
                    minStep=1e-4,
                    numberOfIterations=100,
                    relaxationFactor=0.5,
                    gradientMagnitudeTolerance=1e-8,
                    estimateLearningRate=registration_method.EachIteration,
                )
                registration_method.SetOptimizerScalesFromPhysicalShift()

                # Setup for the multi-resolution framework
                registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
                registration_method.SetSmoothingSigmasPerLevel(
                    smoothingSigmas=[2, 1, 0]
                )
                registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

                # Don't optimize in-place, we would possibly like to run this cell multiple times
                registration_method.SetInitialTransform(
                    initial_transform, inPlace=False
                )
                print("First registering")
                start = time.time()
                # Run the registration
                final_transform = registration_method.Execute(
                    sitk.Cast(fixed_image, sitk.sitkFloat32),
                    sitk.Cast(moving_image, sitk.sitkFloat32),
                )

                print(f"Time to register: {time.time() - start}")

                print("Applying first transformation")
                start = time.time()

                # Now, apply the transform to the top volume overlaping reggion
                resampler = sitk.ResampleImageFilter()
                resampler.SetTransform(final_transform)

                # Set the properties of the resampler to match the original top volume
                resampler.SetOutputSpacing(fixed_image_full.GetSpacing())
                resampler.SetSize(fixed_image_full.GetSize())
                resampler.SetOutputDirection(fixed_image_full.GetDirection())
                resampler.SetOutputOrigin(fixed_image_full.GetOrigin())
                resampler.SetDefaultPixelValue(fixed_image_full.GetPixelIDValue())

                # Apply the transformation
                registered_volume_top_overlapping = resampler.Execute(moving_image_full)

                # Now, apply the transform to the top volume overlaping reggion
                resampler = sitk.ResampleImageFilter()
                resampler.SetTransform(final_transform)

                # Set the properties of the resampler to match the original top volume
                resampler.SetOutputSpacing(final_bot.GetSpacing())
                resampler.SetSize(final_bot.GetSize())
                resampler.SetOutputDirection(final_bot.GetDirection())
                resampler.SetOutputOrigin(final_bot.GetOrigin())
                resampler.SetDefaultPixelValue(final_bot.GetPixelIDValue())

                # Apply the transformation
                final_top_2 = resampler.Execute(final_top)
                print(f"Time to apply transformation: {time.time() - start}")

                print("Creating masks")
                start = time.time()

                mask_top_full = self.mask(
                    sitk.GetArrayFromImage(registered_volume_top_overlapping)
                )
                moving_mask_image_full = sitk.GetImageFromArray(mask_top_full * 255)

                # Initial alignment of the two volumes
                initial_transform2 = sitk.CenteredTransformInitializer(
                    fixed_image_full,
                    registered_volume_top_overlapping,
                    sitk.AffineTransform(3),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY,
                )
                # Set up the registration framework
                registration_method = sitk.ImageRegistrationMethod()

                # mask settings
                registration_method.SetMetricFixedMask(fixed_mask_image_full)
                registration_method.SetMetricMovingMask(moving_mask_image_full)

                # Similarity metric settings
                registration_method.SetMetricAsMattesMutualInformation(
                    numberOfHistogramBins=50
                )
                # registration_method.SetMetricAsMeanSquares()
                registration_method.SetMetricSamplingStrategy(
                    registration_method.RANDOM
                )
                registration_method.SetMetricSamplingPercentage(0.2)

                # Interpolator
                registration_method.SetInterpolator(sitk.sitkLinear)

                # Optimizer settings
                registration_method.SetOptimizerAsRegularStepGradientDescent(
                    learningRate=2.0,
                    minStep=1e-4,
                    numberOfIterations=100,
                    relaxationFactor=0.5,
                    gradientMagnitudeTolerance=1e-8,
                    estimateLearningRate=registration_method.EachIteration,
                )
                registration_method.SetOptimizerScalesFromPhysicalShift()

                # Setup for the multi-resolution framework
                registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
                registration_method.SetSmoothingSigmasPerLevel(
                    smoothingSigmas=[2, 1, 0]
                )
                registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

                # Don't optimize in-place, we would possibly like to run this cell multiple times
                registration_method.SetInitialTransform(
                    initial_transform2, inPlace=False
                )
                print(f"Time to create masks: {time.time() - start}")
                print("Second registering")
                start = time.time()
                # Run the registration
                final_transform2 = registration_method.Execute(
                    sitk.Cast(fixed_image_full, sitk.sitkFloat32),
                    sitk.Cast(registered_volume_top_overlapping, sitk.sitkFloat32),
                )
                print(f"Time to register: {time.time() - start}")

                print("Applying second transformation")
                # Now, apply this transform to the original top volume
                resampler = sitk.ResampleImageFilter()
                resampler.SetTransform(final_transform2)

                # Set the properties of the resampler to match the original top volume
                resampler.SetOutputSpacing(final_bot.GetSpacing())
                resampler.SetSize(final_bot.GetSize())
                resampler.SetOutputDirection(final_bot.GetDirection())
                resampler.SetOutputOrigin(final_bot.GetOrigin())
                resampler.SetDefaultPixelValue(0)

                # Apply the transformation
                registered_volume_top = resampler.Execute(final_top_2)

                # Check if a folder was selected
                if save_path:
                    # Create a progress window

                    # Save the centered volume
                    self.update_progress(0, "Saving registered Volume")
                    image_sequence.write_sequence2(
                        save_path,
                        os.path.basename(save_path),
                        sitk.GetArrayFromImage(registered_volume_top),
                        progress_window=self,
                    )

                    # Show a message box if the centered volume is saved successfully, when the message box is closed, close all matplotlib figures
                    self.update_progress(100, "Saving registered Volume", 1, 1)
                    self.prompt_message("Registered volume saved successfully.")

    def get_roi(self, vol):
        shape = vol.shape

        center_x = shape[1] // 2
        center_y = shape[2] // 2

        rectangle_height = int(shape[1] * 0.08)
        rectangle_width = int(shape[2] * 0.3)

        x0 = center_x - rectangle_height
        x1 = center_x + rectangle_height
        y0 = center_y - rectangle_width
        y1 = center_y + rectangle_width

        return [x0, x1, y0, y1]

    def get_info(self, vol1shape, vol2shape, show=False):
        # ask the user for length(int), resolution (decimal) and frameid(int)
        length, ok = QInputDialog.getInt(
            self.main_window, "Length", "Enter the length of the sample in millimeters:"
        )
        resolution, ok = QInputDialog.getDouble(
            self.main_window,
            "Resolution",
            "Enter the resolution of the scan in millimeters:",
            decimals=5,
        )
        # calculate overlapping region
        lengtha = resolution * vol1shape

        lengthb = resolution * vol2shape

        end_top = length - lengtha

        slice1 = int(end_top / resolution)

        slice2 = int((lengtha - end_top) / resolution)

        return [slice1, slice2]

    def get_roi_2(self, volume):
        from utils.volume_utils import virtual_sequence_bbox as vsb

        # get the bbox of the volume
        bbox = vsb(volume)
        # get the roi of the volume
        return bbox[1]

    def mask(self, volume, top_threshold=235):
        vol = volume.copy()

        vol[vol > top_threshold] = 0

        # Otsu's thresholding
        thresh = threshold_otsu(vol)
        binary = vol > thresh

        binary = binary.astype(np.uint8)

        # label
        labeled = label(binary)

        # regionprops
        props = regionprops(labeled)

        # Find the largest connected component
        largest_component = max(props, key=lambda region: region.area)
        # Create a mask to keep only the largest component
        mask = np.zeros_like(labeled)
        mask[labeled == largest_component.label] = 1
        mask = binary_fill_holes(mask).astype(int)

        return mask * 255
