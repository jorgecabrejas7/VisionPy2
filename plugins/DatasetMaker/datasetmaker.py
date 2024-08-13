import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
from skimage import measure
from skimage.measure import regionprops
import pandas as pd
from skimage.util import view_as_windows


def divide_into_patches(image, patch_size, step_size):
    patches = view_as_windows(image, (image.shape[0], patch_size, patch_size), step=(image.shape[0], step_size, step_size))
    return patches.reshape(-1, image.shape[0], patch_size, patch_size)

def calculate_patch_shape(image_shape, patch_size, step_size):
    # Calculate the number of patches along each dimension
    num_patches_h = ((image_shape[1] - patch_size) // step_size) + 1
    num_patches_w = ((image_shape[2] - patch_size) // step_size) + 1

    # The outgoing shape would be (num_patches_h, num_patches_w, image_shape[0], patch_size, patch_size)
    return (num_patches_h * num_patches_w, image_shape[0], patch_size, patch_size)


def calculate_pixels(ut_resolution, xct_resolution, ut_pixels):
    # Calculate the ratio of the resolutions
    resolution_ratio = ut_resolution /xct_resolution 
    
    # Calculate the equivalent number of pixels in the xct resolution
    xct_pixels = ut_pixels * resolution_ratio
    
    return xct_pixels

def crop_image_to_patch_size(image, patch_size):
    z, x, y = image.shape

    crop_x = x % patch_size
    crop_y = y % patch_size

    crop_x_before = crop_x // 2
    crop_x_after = crop_x - crop_x_before

    crop_y_before = crop_y // 2
    crop_y_after = crop_y - crop_y_before

    cropped_image = image[:, crop_x_before:-crop_x_after or None, crop_y_before:-crop_y_after or None]

    return cropped_image

def nearest_lower_divisible(num, divisor):
	if num % divisor == 0:
		return num
	else:
		remainder = num % divisor
		return int(num - remainder)

def nearest_higher_divisible(num, divisor):
    if num % divisor == 0:
        return num
    else:
        remainder = num % divisor
        return int(num + divisor - remainder)
    
def nearest_bounding_box(minr, minc, maxr, maxc, divisor):
    minr = nearest_higher_divisible(minr, divisor)
    minc = nearest_higher_divisible(minc, divisor)
    maxr = nearest_lower_divisible(maxr, divisor)
    maxc = nearest_lower_divisible(maxc, divisor)

    return minr, minc, maxr, maxc

def preprocess(onlypores,mask,ut_rf,ut_amp, xct_resolution = 0.025, ut_resolution = 1):

    #Cropeamos onlypores y la mascara para quitarnos el fondo y le aplicamos la misma bounding box a los datos de UT

    # Calculate scaling factor
    scaling_factor = xct_resolution / ut_resolution

    #XCT 

    max_proj = np.max(mask, axis=0)

    labels = measure.label(max_proj)

    props = regionprops(labels)

    minr_xct, minc_xct, maxr_xct, maxc_xct = props[0].bbox

    minr_xct, minc_xct, maxr_xct, maxc_xct = nearest_bounding_box(minr_xct, minc_xct, maxr_xct, maxc_xct, 1/scaling_factor)

    #crop the volume

    mask_cropped = mask[:, minr_xct:maxr_xct, minc_xct:maxc_xct]
    onlypores_cropped = onlypores[:, minr_xct:maxr_xct, minc_xct:maxc_xct]

    #UT

    # Convert bounding box to UT resolution
    minr_ut = int(minr_xct * scaling_factor)
    minc_ut = int(minc_xct * scaling_factor)
    maxr_ut = int(maxr_xct * scaling_factor)
    maxc_ut = int(maxc_xct * scaling_factor)

    ut_rf_cropped = ut_rf[:,minr_ut:maxr_ut, minc_ut:maxc_ut]
    ut_amp_cropped = ut_amp[:,minr_ut:maxr_ut, minc_ut:maxc_ut]

    return onlypores_cropped, mask_cropped, ut_rf_cropped, ut_amp_cropped

def patch(onlypores_cropped, mask_cropped, ut_rf_cropped, ut_amp_cropped, ut_patch_size = 3, ut_step_size =1, xct_resolution = 0.025, ut_resolution = 1):

    #compute xct patch size
    ut_patch_size = 3
    ut_step_size = 1
    xct_patch_size = int(np.round(calculate_pixels(ut_resolution, xct_resolution, ut_patch_size)))
    xct_step_size = int(np.round(calculate_pixels(ut_resolution, xct_resolution, ut_step_size)))

    #crop volumes to fit patch size division
    ut_amp_cropped = crop_image_to_patch_size(ut_amp_cropped, ut_patch_size)
    ut_rf_cropped = crop_image_to_patch_size(ut_rf_cropped, ut_patch_size)
    onlypores_cropped = crop_image_to_patch_size(onlypores_cropped, xct_patch_size)
    mask_cropped = crop_image_to_patch_size(mask_cropped, xct_patch_size)

    #ensure patches are the same
    ut_shape = calculate_patch_shape(ut_rf_cropped.shape, ut_patch_size, ut_step_size)
    xct_shape = calculate_patch_shape(onlypores_cropped.shape, xct_patch_size, xct_step_size)

    print(ut_shape[0], xct_shape[0])

    if not (ut_shape[0] == xct_shape[0]):
        print('Patches are not the same')
        return 0,0,0,0
    
    #divide into patches
    patches_ut = divide_into_patches(ut_rf_cropped, ut_patch_size, ut_step_size)
    patches_ut_amp = divide_into_patches(ut_amp_cropped, ut_patch_size, ut_step_size)

    patches_onlypores = divide_into_patches(onlypores_cropped, xct_patch_size, xct_step_size)
    patches_mask = divide_into_patches(mask_cropped,xct_patch_size, xct_step_size)

    return patches_onlypores, patches_mask, patches_ut, patches_ut_amp 

def datasets1_2(patches_onlypores, patches_mask, patches_ut, patches_ut_amp, folder):

    #compute the sum of onlypores and mask
    sum_onlypores_patches = np.sum(patches_onlypores, axis = 1)
    sum_mask_patches = np.sum(patches_mask, axis = 1)

    #######volfrac for patch vs volfrac dataset

    sum_onlypores = np.sum(sum_onlypores_patches, axis = (1,2))
    sum_mask = np.sum(sum_mask_patches, axis = (1,2))

    #the points that are zero in the mask are not material, so we set them to -1 in volfrac to know that they are not material

    zero_indices = np.where(sum_mask == 0)

    volfrac = sum_onlypores / (sum_mask + 1e-6)

    volfrac[zero_indices] = -1

    ########volfrac for patch vs volfrac patch dataset

    zero_indices = np.where(sum_mask_patches == 0)

    volfrac_patches = sum_onlypores_patches / (sum_mask_patches + 1e-6)

    volfrac_patches[zero_indices] = -1

    volfrac_patches = volfrac_patches.reshape(-1, volfrac_patches.shape[1] * volfrac_patches.shape[2])

    #prepare ut for dataframe

    ut_patches_reshaped = patches_ut.transpose(0, 2, 3, 1)

    ut_amp_patches_reshaped = patches_ut_amp.transpose(0, 2, 3, 1)

    ut_patches_reshaped = ut_patches_reshaped.reshape(ut_patches_reshaped.shape[0], -1)

    ut_amp_patches_reshaped = ut_amp_patches_reshaped.reshape(patches_ut_amp.shape[0], -1)

    combined_ut = np.hstack((ut_patches_reshaped, ut_amp_patches_reshaped))

    #create both dataframes

    #column names for ut

    columns_ut = []

    half = combined_ut.shape[1] // 2

    for i in range(combined_ut.shape[1]):
        if i < half:
            columns_ut.append(f'ut_rf_{i}')
        else:
            columns_ut.append(f'ut_amp_{i - half}')

    columns_ut = np.array(columns_ut)

    #dataframe for patch vs volfrac dataset

    patch_vs_volfrac = np.hstack((combined_ut, volfrac.reshape(-1,1)))

    df_patch_vs_volfrac = pd.DataFrame(patch_vs_volfrac, columns = np.append(columns_ut, 'volfrac'))

    
    #column names for volfrac patches dataframes
    columns_volfrac = []

    for i in range(volfrac_patches.shape[1]):
        columns_volfrac.append(f'volfrac_{i}')

    columns_volfrac = np.array(columns_volfrac)

    #dataframe for patch vs volfrac patch dataset

    patch_vs_patch = np.hstack((combined_ut, volfrac_patches))

    columns = np.append(columns_ut, columns_volfrac)

    df_patch_vs_patch = pd.DataFrame(patch_vs_patch, columns = columns)

    #save the dataframes

    df_patch_vs_volfrac.to_csv(folder / 'patch_vs_volfrac.csv', index = False)

    df_patch_vs_patch.to_csv(folder / 'patch_vs_patch.csv', index = False)

def main(onlypores,mask,ut_rf,ut_amp,folder, xct_resolution = 0.025, ut_resolution = 1,ut_patch_size = 3, ut_step_size =1):

    print('Preprocessing and patching the images...')
    #preprocess the images
    onlypores_cropped, mask_cropped, ut_rf_cropped, ut_amp_cropped = preprocess(onlypores,mask,ut_rf,ut_amp, xct_resolution, ut_resolution)
    print('Patching the images...')
    #patch the images
    patches_onlypores, patches_mask, patches_ut, patches_ut_amp = patch(onlypores_cropped, mask_cropped, ut_rf_cropped, ut_amp_cropped, ut_patch_size, ut_step_size, xct_resolution, ut_resolution)
    print('Creating the datasets...')
    #create the datasets
    datasets1_2(patches_onlypores, patches_mask, patches_ut, patches_ut_amp, folder)


    

