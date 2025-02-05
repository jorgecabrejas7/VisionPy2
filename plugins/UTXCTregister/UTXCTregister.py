import tifffile as tiff
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import SimpleITK as sitk
from skimage.filters import threshold_otsu
from scipy.ndimage import minimum_filter, label, binary_fill_holes
from skimage.measure import regionprops
import cv2
import scipy.ndimage
from joblib import Parallel, delayed


def dilate_image(image, kernel_size=5, iterations=1):
    """
    Dilates an image using a square kernel of a given size and number of iterations.
    
    Parameters:
    - image: Input image.
    - kernel_size: Size of the square kernel. Default is 5.
    - iterations: Number of times dilation is applied. Default is 1.
    
    Returns:
    - Dilated image.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)
    return dilated_image

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

def find_holes_minimums(image, labeled_image):
    """
    Find the center of the minimum value in each labeled region of the image.
    
    Args:
    image (numpy.ndarray): 8-bit input image.
    labeled_image (numpy.ndarray): Labeled image with regions.
    
    Returns:
    numpy.ndarray: Array of centroids of the minimum values in each region.
    """
    regions = regionprops(labeled_image, intensity_image=image)
    centers = []

    for region in regions:
        # Get the coordinates of the region
        coords = region.coords
        # Get the intensity values of the region
        intensities = image[coords[:, 0], coords[:, 1]]
        # Find the minimum intensity value
        min_value = np.min(intensities)
        # Get the coordinates of the minimum intensity value
        min_coords = coords[intensities == min_value]
        
        if len(min_coords) > 1:
            # If there are multiple minimums, find the most centered one
            center = np.mean(min_coords, axis=0)
            distances = np.linalg.norm(min_coords - center, axis=1)
            min_coords = min_coords[np.argmin(distances)]
        else:
            min_coords = min_coords[0]
        
        centers.append(min_coords[::-1])
    
    return np.array(centers)

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

def extract_points(image): #given a labeled points image, returns the list of points in it x,y,value
    points = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] > 0:
                points.append([i,j,image[i,j]])
    return np.array(points)

def rigid_body_transformation_matrix(points_A, points_B):
    """
    Calculate the rigid body transformation matrix to convert points_B to points_A.
    
    Args:
    points_A (numpy.ndarray): A Nx2 array of points.
    points_B (numpy.ndarray): A Nx2 array of points.
    
    Returns:
    numpy.ndarray: A 3x3 transformation matrix.
    """
    
    # Compute the centroids of both sets of points
    centroid_A = np.mean(points_A, axis=0)
    centroid_B = np.mean(points_B, axis=0)
    
    # Center the points by subtracting the centroids
    centered_A = points_A - centroid_A
    centered_B = points_B - centroid_B
    
    # Compute the covariance matrix
    H = np.dot(centered_B.T, centered_A)
    
    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)
    
    # Compute the rotation matrix
    R = np.dot(Vt.T, U.T)
    
    # Ensure a right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    # Compute the translation vector
    t = centroid_A - np.dot(R, centroid_B)
    
    # Construct the transformation matrix
    transformation_matrix = np.eye(3)
    transformation_matrix[:2, :2] = R
    transformation_matrix[:2, 2] = t
    
    return transformation_matrix

def scale_transformation_matrix(transformation_matrix, scale_factor):
    """
    Scale a 2D rigid body transformation matrix.
    
    Args:
    transformation_matrix (numpy.ndarray): A 3x3 transformation matrix.
    scale_factor (float): Scaling factor.
    
    Returns:
    numpy.ndarray: A 3x3 scaled transformation matrix.
    """
    
    # Construct the scaling matrix
    scaled_transformation_matrix = transformation_matrix.copy()

    scaled_transformation_matrix[:2,2] *= scale_factor


    return scaled_transformation_matrix

def xct_preprocessing(xct,original_resolution=0.025,new_resolution=1):

    #get the max projection of the xct so the holes are visible
    max_proj = np.max(xct, axis=2)

    #resize the brightest to UT resolution
    original_dimensions = (xct.shape[0], xct.shape[1])
    new_dimensions = calculate_new_dimensions(original_resolution, new_resolution, original_dimensions)

    #segment and get onlypores to get only the holes
    thresh = threshold_otsu(max_proj)
    binary = max_proj > thresh

    mask = binary_fill_holes(binary)

    inverted = np.invert(binary)

    circles = np.logical_and(mask, inverted)

    circles = resize_image(Image.fromarray(circles), new_dimensions[::-1])

    xct_centers  = find_rectangle_centers(circles)

    centers_painted_xct = paint_binary_points(circles.shape, xct_centers)

    return centers_painted_xct

def ut_preprocessing(ut):
    
    #get the frontwall slice id
    frontwall = find_brightest_ut(ut)

    #get the frontwall slice
    ut_max_proj = np.max(ut[:,:,frontwall:], axis=2)

    #otsu threshold
    thresh = threshold_otsu(ut_max_proj)
    binary = ut_max_proj > thresh

    mask = binary_fill_holes(binary)

    inverted = np.invert(binary)

    #onlypores to get the holes of the samples
    circles = np.logical_and(mask, inverted)

    #now we label the onlypores to get only the 3 biggest ones

    #labeling
    labeled, _= label(circles)

    #regionprops
    props = regionprops(labeled)

    # Step 1: Sort regions by area in descending order and get their labels
    sorted_regions = sorted(props, key=lambda x: x.area, reverse=True)
    largest_regions_labels = [region.label for region in sorted_regions[:3]]

    # Step 2: Create a new image to hold the result
    largest_regions_image = np.zeros_like(labeled)

    # Step 3: Fill in the three largest regions
    for lbl in largest_regions_labels:
        largest_regions_image[labeled == lbl] = lbl

    ut_centers  = find_rectangle_centers(largest_regions_image > 0)

    centers_painted_ut = paint_binary_points(largest_regions_image.shape, ut_centers)

    return centers_painted_ut

def ut_preprocessing_2(ut):

    #get the frontwall slice
    ut_max_proj = np.max(ut[:,:,:], axis=2)

    #otsu threshold
    thresh = threshold_otsu(ut_max_proj)
    binary = ut_max_proj > thresh

    mask = binary_fill_holes(binary)

    inverted = np.invert(binary)

    #onlypores to get the holes of the samples
    circles = np.logical_and(mask, inverted)

    #now we label the onlypores to get only the 3 biggest ones

    #labeling
    labeled, _= label(circles)

    #regionprops
    props = regionprops(labeled)

    # Step 1: Sort regions by area in descending order and get their labels
    sorted_regions = sorted(props, key=lambda x: x.area, reverse=True)
    largest_regions_labels = [region.label for region in sorted_regions[:3]]

    # Step 2: Create a new image to hold the result
    largest_regions_image = np.zeros_like(labeled)
    labeled_image = np.zeros_like(labeled)

    # Step 3: Fill in the three largest regions
    for lbl in largest_regions_labels:
        largest_regions_image[labeled == lbl] = ut_max_proj[labeled == lbl]
        labeled_image[labeled == lbl] = lbl

    ut_centers  = find_holes_minimums(largest_regions_image, labeled_image)

    centers_painted_ut = paint_binary_points(largest_regions_image.shape, ut_centers)

    return centers_painted_ut

def register(labeled_ut, labeled_xct, save_path):

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

    # registration_method.SetMetricAsMeanSquares()

    #add the masks
    # registration_method.SetMetricFixedMask(fixed_mask)
    # registration_method.SetMetricMovingMask(moving_mask)

    # Optimizer settings
    registration_method.SetOptimizerAsExhaustive(numberOfSteps=[5,5,5], stepLength = 1.0)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework
    # registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4, 2, 1])
    # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    
    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                                                sitk.Cast(moving_image, sitk.sitkFloat32))
    
    save_point_images(final_transform,labeled_ut,labeled_xct,save_path)
    
    #Convert the transform to the original resolution and to a 3D spacing

    # Get the original parameters
    original_parameters = np.array(final_transform.GetParameters())

    transform_3d = sitk.Euler3DTransform()

    # Create a new set of parameters for the 3D transform
    # Set the rotation parameters to the same as the 2D transform
    # Set the translation parameters to the same as the 2D transform for x and y, and 0 for z
    params_3d2 = (original_parameters[0], 0, 0, original_parameters[1], original_parameters[2], 0)

    transform_3d.SetParameters(params_3d2) 

    return transform_3d

def save_point_images(transform,ut_centers,xct_centers,save_path):

    folder = Path(save_path).parent / 'registration_auxiliary_files'

    if not folder.exists():
        folder.mkdir()

    #save ut_centers and xct_centers in the parent folder of save_path
    ut_centers_path = folder / 'ut_centers.tif'
    xct_centers_path = folder / 'xct_centers.tif'
    ut_centers_path_original = folder / 'ut_centers_original.tif'
    xct_centers_path_original = folder / 'xct_centers_original.tif'

    tiff.imsave(ut_centers_path_original, ut_centers)
    tiff.imsave(xct_centers_path_original, xct_centers)

    #apply transformation to xct_centers
    # Now, apply the transform to the top volume overlaping reggion
    fixed_image = sitk.GetImageFromArray(ut_centers)
    moving_image = sitk.GetImageFromArray(xct_centers)
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(0) 

    # Set the properties of the resampler to match the original top volume
    resampler.SetOutputSpacing(fixed_image.GetSpacing())
    resampler.SetSize(fixed_image.GetSize())
    resampler.SetOutputDirection(fixed_image.GetDirection())
    resampler.SetOutputOrigin(fixed_image.GetOrigin())

    # Apply the transformation
    xct_centers = resampler.Execute(moving_image)
    xct_centers = sitk.GetArrayFromImage(xct_centers)

    tiff.imsave(ut_centers_path, ut_centers)
    tiff.imsave(xct_centers_path, xct_centers)

def apply_transform(transform, ut, xct, original_resolution=0.025):

    #Prepare the ut and xct files to be resampled
    new_shape = calculate_new_dimensions(1, original_resolution, (ut.shape[0], ut.shape[1]))

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

def apply_transform_parameters(matrix, ut, xct, original_resolution=0.025): #(X,Y,Z) axes
    big_shape = calculate_new_dimensions(1,original_resolution,ut.shape[:2])

    transformed_volume = []

    if len(xct.shape) == 3:

        for i in range(xct.shape[2]):
            transformed_volume.append(scipy.ndimage.affine_transform(xct[:,:,i], matrix[:2,:], output_shape=big_shape))
            transformed_volume = np.array(transformed_volume)

            transformed_volume = np.swapaxes(transformed_volume, 0, 1)
            transformed_volume = np.swapaxes(transformed_volume, 1, 2)

            return transformed_volume
    else:

        for i in range(2):

            transformed_volume.append(scipy.ndimage.affine_transform(xct, matrix[:2,:], output_shape=big_shape))

            transformed_volume = np.array(transformed_volume)

            transformed_volume = np.swapaxes(transformed_volume, 0, 1)
            transformed_volume = np.swapaxes(transformed_volume, 1, 2)

            return transformed_volume[:,:,0]


def apply_transform_parameters_paralel(matrix, ut, xct, original_resolution=0.025):

    big_shape = calculate_new_dimensions(1,original_resolution,ut.shape[:2])

    def func(slice, matrix, shape):
    # Apply affine transform to the current slice
        return scipy.ndimage.affine_transform(slice, matrix, output_shape=shape)

    # Create the function to apply on each slice
    def process_slice(z, volume, matrix, shape):
        current_slice = volume[:, :, z]  # Extract the (X, Y) slice
        return func(current_slice, matrix, shape)  # Apply the transformation

    # Use joblib's Parallel to apply the function concurrently with a progress bar
    n_jobs = -1  # Use all available cores
    n_slices = xct.shape[2]  # Number of slices in the Z dimension

    # Use tqdm for progress bar
    transformed_slices = Parallel(n_jobs=n_jobs)(
        delayed(process_slice)(z, xct, matrix[:2,:], big_shape) for z in range(n_slices)
    )

    # Reassemble the transformed slices into the final 3D array
    transformed_volume = np.stack(transformed_slices, axis=2)

    return transformed_volume

def get_rotation_angle(transformation_matrix):
    """
    Extract the rotation angle from the transformation matrix.
    
    Args:
    transformation_matrix (numpy.ndarray): A 3x3 transformation matrix.
    
    Returns:
    float: The rotation angle in radians.
    """
    R = transformation_matrix[:2, :2]
    angle = np.arctan2(R[1, 0], R[0, 0])
    return angle


def main_old(ut,xct, path):

    print('Preprocessing')

    #preprocess the files
    ut_centers = ut_preprocessing(ut)
    xct_centers = xct_preprocessing(xct)

    #label the images
    ut_labeled = label_objects(ut_centers)
    xct_labeled = label_objects(xct_centers)

    print('Preprocessed')

    print('Registering')

    #register
    transformation = register(ut_labeled, xct_labeled, path)

    print('Registered')

    print('Applying transformation')

    #apply the transformation
    xct_aligned = apply_transform(transformation, ut, xct)

    print('Transformation applied')

    return xct_aligned

def main(ut,rf,xct,padded=False):

    if not padded:

        pad_shape = (int(ut.shape[0]/0.025), int(ut.shape[1]/0.025))

        pad_diff = (pad_shape[0] - xct.shape[0], pad_shape[1] - xct.shape[1])

        xct = np.pad(xct, ((pad_diff[0]//2, pad_diff[0]-pad_diff[0]//2), (pad_diff[1]//2, pad_diff[1]-pad_diff[1]//2), (0, 0)), mode='constant', constant_values=0)

    print('Preprocessing')

    #preprocess the files
    ut_centers = ut_preprocessing(ut)
    xct_centers = xct_preprocessing(xct)

    #label the images
    ut_labeled = label_objects(ut_centers)
    xct_labeled = label_objects(xct_centers)

    #extract the points from the images
    ut_points = extract_points(ut_labeled)
    xct_points = extract_points(xct_labeled)

    sorted_indices_ut = np.argsort(ut_points[:, -1])
    sorted_indices_xct = np.argsort(xct_points[:, -1])

    # Use these indices to sort the original array
    sorted_ut_points = ut_points[sorted_indices_ut]
    sorted_xct_points = xct_points[sorted_indices_xct]

    #check if there are at least 2 points
    if len(sorted_ut_points) < 3:
        print('Not enough UT points')
        raise ValueError('Not enough UT points')

    if len(sorted_xct_points) < 3:
        print('Not enough XCT points')
        raise ValueError('Not enough XCT points')

    #check if there are the same number of points
    if len(sorted_ut_points) != len(sorted_xct_points):
        print('Different number of points')
        raise ValueError('Different number of points')

    print('Preprocessed')

    print('Registering')

    transformation_matrix = rigid_body_transformation_matrix(sorted_ut_points[:,:2],sorted_xct_points[:,:2])

    print('Registered')

    print('Applying transformation')

    transformed_volume = []

    for i in range(rf.shape[2]):
        transformed_volume.append(scipy.ndimage.affine_transform(rf[:,:,i], transformation_matrix[:2,:], output_shape=xct_centers.shape))

    transformed_volume = np.array(transformed_volume)

    print('Transformation applied')

    if not padded:

        return (transformed_volume,xct)

    return transformed_volume


# new register

def main_2(ut,xct):

    print('Preprocessing')

    ut_centers = label_objects(ut_preprocessing_2(ut))

    xct_centers = label_objects(xct_preprocessing(xct))

    #extract the points from the images
    ut_points = extract_points(ut_centers)
    xct_points = extract_points(xct_centers)

    sorted_indices_ut = np.argsort(ut_points[:, -1])
    sorted_indices_xct = np.argsort(xct_points[:, -1])

    # Use these indices to sort the original array
    sorted_ut_points = ut_points[sorted_indices_ut]
    sorted_xct_points = xct_points[sorted_indices_xct]

    #check if there are at least 2 points
    if len(sorted_ut_points) < 3:
        print('Not enough UT points')
        raise ValueError('Not enough UT points')

    if len(sorted_xct_points) < 3:
        print('Not enough XCT points')
        raise ValueError('Not enough XCT points')

    #check if there are the same number of points
    if len(sorted_ut_points) != len(sorted_xct_points):
        print('Different number of points')
        raise ValueError('Different number of points')
    
    print('Preprocessed')

    print('Registering')

    transformation_matrix = rigid_body_transformation_matrix(sorted_xct_points[:,:2],sorted_ut_points[:,:2])
    scaled_transformation_matrix = scale_transformation_matrix(transformation_matrix, 1/0.025)

    print('Registered')

    parameters = scaled_transformation_matrix

    #apply the original transformation to the xct centers
    transformed_xct_centers = apply_transform_parameters(transformation_matrix,ut,(xct_centers > 0) * 255, original_resolution=1)

    return parameters,ut_centers,xct_centers, transformed_xct_centers

def apply_registration(ut,xct,parameters): #(X,Y,Z) axes

    transformation_matrix = parameters

    print('Applying transformation')

    transformed_volume = apply_transform_parameters_paralel(transformation_matrix,ut,xct)

    print('Transformation applied')

    return transformed_volume