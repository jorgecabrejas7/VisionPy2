import tifffile as tiff
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
import SimpleITK as sitk
from skimage.filters import threshold_otsu
from scipy.ndimage import minimum_filter, label

def resize_image(original_image, size, show = False):
    width, height = original_image.size
    if show:
        print(f"The original image size is {width} wide x {height} tall")

    resized_image = original_image.resize(size)
    width, height = resized_image.size
    if show:
        print(f"The resized image size is {width} wide x {height} tall")
    return np.array(resized_image)

def calculate_new_dimensions(original_resolution, new_resolution, original_dimensions):
    # Calculate the original dimensions in real-world units
    original_width, original_height = original_dimensions
    real_world_width = original_width * original_resolution
    real_world_height = original_height * original_resolution

    # Calculate the new dimensions in pixels
    new_width = int(real_world_width / new_resolution)
    new_height = int(real_world_height / new_resolution)

    return new_width, new_height

def get_brightest(d):
    data = d.copy()
    #turn to 0 the values below 255
    data[data < 255] = 0
    #calculate the max of each image
    max_values = np.sum(data, axis=(0, 1))
    
    return np.argmax(max_values)

def paste_image_center(img1_array, img2_array):
    # Convert numpy arrays to PIL Images
    img1 = Image.fromarray(img1_array)
    img2 = Image.fromarray(img2_array)

    # Create a new image of zeros with the same shape as img1
    new_img = Image.fromarray(np.zeros_like(img1_array))

    # Calculate the position for the second image
    position = ((new_img.size[0]-img2.size[0])//2, (new_img.size[1]-img2.size[1])//2)

    # Paste the second image onto the new image
    new_img.paste(img2, position)

    # Convert back to numpy array and return
    return np.array(new_img)

def label_objects(image):
    labeled_image, num_features = label(image)
    output_image = np.zeros_like(image)
    
    indices = np.where(labeled_image > 0)
    indices = list(zip(indices[0], indices[1]))
    
    # Label the object nearest to the top edge
    top_object = min(indices, key=lambda x: x[0])
    output_image[top_object] = 100
    indices.remove(top_object)
    
    if indices:
        # Label the object nearest to the left edge
        left_object = min(indices, key=lambda x: x[1])
        output_image[left_object] = 175
        indices.remove(left_object)
    
    # Label the remaining objects
    for idx in indices:
        output_image[idx] = 255
    
    return output_image

def plot_images(images, figsz = (5, 5)):
    fig, axs = plt.subplots(1, len(images), figsize=figsz)
    for i, img in enumerate(images):
        axs[i].imshow(img, cmap='gray')
        axs[i].axis('off')
    plt.show()

def circles(img):
    # Convert binary image to grayscale
    img = (img * 255).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty image to draw circles on
    circle_img = np.zeros_like(img)

    # For each contour
    for contour in contours:
        # Find minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        # Draw the circle
        cv2.circle(circle_img, (int(x), int(y)), int(radius), (255, 255, 255), -1)

        break

    return circle_img

def rectangles(img, thickness=2):
    # Convert binary image to grayscale
    img = (img * 255).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty image to draw rectangles on
    rectangle_img = np.zeros_like(img)

    # For each contour
    for contour in contours:
        # Find bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        # Draw the rectangle contour
        cv2.rectangle(rectangle_img, (x, y), (x+w, y+h), (255, 255, 255), thickness)

    return rectangle_img > 0

import cv2
import numpy as np

def draw_bounding_box(image):

    # Convert binary image to grayscale
    image = (image * 255).astype(np.uint8)

    # Create a new binary image of the same size
    output_image = np.zeros_like(image)

    # Find contours in the image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, draw the bounding box on the new image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output_image, (x, y), (x+w, y+h), (255), -1)

    return output_image > 0

def find_rectangle_centers(image):
    # Ensure the image is binary
    assert np.array_equal(image, image.astype(bool)), "Image must be binary"

    # Find connected components in the image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image.astype('uint8'))

    # The first component is the background, so ignore it
    return centroids[1:]

def paint_points_on_image(points, image):
    # Create a copy of the image to avoid modifying the original
    image_copy = np.copy(image)

    # Convert the image to RGB if it's grayscale
    if len(image_copy.shape) == 2:
        image_copy = np.stack((image_copy,)*3, axis=-1)

    # Paint each point in red
    for point in points:
        x, y = point
        image_copy[int(y), int(x)] = [255, 0, 0]  # RGB for red

    return image_copy

def find_local_minima(image, size=3, threshold=50):
    # Apply the minimum filter
    filtered_image = minimum_filter(image, size)

    # Find local minima
    local_minima = image == filtered_image

    # Apply threshold
    local_minima = np.logical_and(local_minima, image <= threshold)

    # The result is a boolean mask where True indicates local minima
    return local_minima

def find_brightest_ut(volume):
    # Find the brightest slice
    brightest_slice = np.argmax(np.sum(volume, axis=(0, 1)))

    return brightest_slice

def paint_binary_points(shape, points):
    # Create an empty image of the specified shape
    image = np.zeros(shape, dtype=np.uint8)

    # Iterate over the points
    for point in points:
        # Round the coordinates to the nearest integer
        y,x = tuple(int(round(coord)) for coord in point)
        # cv2.circle(image, (y,x), 5, (255), -1)
        # Draw the point on the image
        image[x,y] = 255

    return image.astype(np.uint8)

def xct_preprocessing(xct,original_resolution=0.022,new_resolution=1):


    #get the brightest slice, the one with the stickers in it
    id = get_brightest(xct)

    #resize the brightest slice to UT resolution
    original_dimensions = (xct.shape[0], xct.shape[1])
    new_dimensions = calculate_new_dimensions(original_resolution, new_resolution, original_dimensions)
    resized = resize_image(Image.fromarray(xct[:,:,id]), new_dimensions[::-1])

    #threshold the image
    thresh = threshold_otsu(resized)
    thresholded = resized > thresh

    #find the centers of the stickers
    xct_centers  = find_rectangle_centers(thresholded)

    centers_painted_xct = paint_binary_points(thresholded.shape, xct_centers)

    return centers_painted_xct

def ut_preprocessing(ut):

    #get the frontwall slice id
    aligned_id = find_brightest_ut(ut)

    #get the frontwall slice
    frontwall = ut[:,:,aligned_id].reshape(ut.shape[0], ut.shape[1])

    #otsu threshold the frontwall
    thresh = threshold_otsu(frontwall)
    thresholded = frontwall > thresh
    #fill the holes
    thresholded = ndimage.binary_fill_holes(thresholded)

    #find the local minimas to get the centers of the stickers
    minimums = find_local_minima(frontwall, 10)

    #erase the minimus out of the thresholded image, to prevent getting minimus out of the sample
    only_minimus = minimums * thresholded

    #get the minimus as a list of points
    centers = find_rectangle_centers(only_minimus)

    #create and image with the points painted
    centers_painted_ut = paint_binary_points(frontwall.shape, centers)

    return centers_painted_ut

def register(labeled_ut, labeled_xct):

    fixed_image = sitk.GetImageFromArray(labeled_ut)
    moving_image = sitk.GetImageFromArray(labeled_xct)
    # Initial alignment of the two volumes
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                        moving_image, 
                                                        sitk.Euler2DTransform(), 
                                                        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    # Set up the registration framework
    registration_method = sitk.ImageRegistrationMethod()
    # Similarity metric settings
    registration_method.SetMetricAsMattesMutualInformation()

    # Optimizer settings
    registration_method.SetOptimizerAsExhaustive(numberOfSteps=[10,10,10], stepLength = 1.0)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    
    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                                                sitk.Cast(moving_image, sitk.sitkFloat32))
    
    #Convert the transform to the original resolution and to a 3D spacing

    # Get the original parameters
    original_parameters = np.array(final_transform.GetParameters())

    # Compute the scaling factor
    scaling_factor = 1 / 0.22

    # Adjust the translation parameters
    adjusted_parameters = original_parameters.copy()
    adjusted_parameters[1:] *= scaling_factor

    transform_3d = sitk.Euler3DTransform()

    # Create a new set of parameters for the 3D transform
    # Set the rotation parameters to the same as the 2D transform
    # Set the translation parameters to the same as the 2D transform for x and y, and 0 for z
    params_3d2 = (original_parameters[0], 0, 0, original_parameters[1], original_parameters[2], 0)

    transform_3d.SetParameters(params_3d2) 

    return transform_3d


def apply_transform(transform, ut, xct, original_resolution=0.022):

    #Prepare the ut and xct files to be resampled
    new_shape = calculate_new_dimensions(1, 0.022, (ut.shape[0], ut.shape[1]))

    big_ut = np.zeros((xct.shape[2],new_shape[0], new_shape[1]), dtype=np.uint8)

    big_ut_itk = sitk.GetImageFromArray(big_ut)
    big_ut_itk.SetSpacing((original_resolution, original_resolution,original_resolution))

    xct = np.swapaxes(xct, 2, 1)
    xct = np.swapaxes(xct, 1, 0)

    xct_itk = sitk.GetImageFromArray(xct)
    xct_itk.SetSpacing((original_resolution, original_resolution,original_resolution))

    #apply the transformation
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)

    resampler.SetOutputSpacing(big_ut_itk.GetSpacing())
    resampler.SetOutputOrigin(big_ut_itk.GetOrigin())
    resampler.SetOutputDirection(big_ut_itk.GetDirection())
    resampler.SetSize(big_ut_itk.GetSize())

    resampled_xct = resampler.Execute(xct_itk)

    resampled_xct = sitk.GetArrayFromImage(resampled_xct).astype(np.uint8)

    return resampled_xct


