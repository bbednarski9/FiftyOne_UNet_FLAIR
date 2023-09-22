'''
First, define a few helper functions to make help with some file conversions.
'''
import os
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm
import tifffile

def parse_img_mask_filenames(all_files, extension):
    """
    Sort the files in the patient subdirectory into image and mask lists.
    """
    imgs, masks = [], []
    mask_extension = '_mask'+extension
    for file in all_files:
        if file.endswith(extension):
            if file.endswith(mask_extension):
                masks.append(file)
            else:
                imgs.append(file)
    return imgs, masks

def sort_filenames_in_series(files, patient_id, extension):
    files_clean = files.copy()
    files_clean = [file.strip(extension) for file in files_clean]
    files_clean = [file.strip('_mask') for file in files_clean]
    files_clean = [file.replace(patient_id+"_", "") for file in files_clean]
    sort_order = np.argsort(files_clean)
    files = [files[i] for i in sort_order]
    return files

def convert_tif_to_npy(old_patient_dir, new_patient_dir, img, return_sample=False):
    fp_tif = os.path.join(old_patient_dir, img)
    # load the .tif file as a numpy array
    img_arr = tifffile.imread(fp_tif)
    # define and use a regular expression to extract the slide name from the filepath
    np.save(os.path.join(new_patient_dir, img.replace('.tif', '.npy')), img_arr)
    if return_sample:
        return img_arr # THIS IS A NUMPY ARRAY
    else:
        return None
    
def convert_npz_to_jpeg(old_patient_dir, new_patient_dir, img, return_sample=False):
    fp_np = os.path.join(old_patient_dir, img)
    # load the .tif file as a numpy array
    img_arr = np.load(fp_np)
    # define and use a regular expression to extract the slide name from the filepath
    img_arr = np.squeeze(img_arr)
    img_pil = Image.fromarray(img_arr)
    img_pil.save(os.path.join(new_patient_dir, img.replace('.npy', '.jpeg')), "JPEG")
    if return_sample:
        return img_pil # THIS IS A PIL IMAGE
    else:
        return None


def convert_npz_to_png(old_patient_dir, new_patient_dir, img, return_sample=False):
    fp_np = os.path.join(old_patient_dir, img)
    # load the .tif file as a numpy array
    img_arr = np.load(fp_np)
    # define and use a regular expression to extract the slide name from the filepath
    img_arr = np.squeeze(img_arr)
    img_pil = Image.fromarray(img_arr)
    img_pil = img_pil.convert('RGB')
    img_pil.save(os.path.join(new_patient_dir, img.replace('.npy', '.png')))
    if return_sample:
        return img_pil # THIS IS A PIL IMAGE
    else:
        return None
    
'''
.tif direcotry to .npy directory
'''
def tif_dirs_to_npz(root_tif_dir, root_npz_dir, patients, return_sample=False):
    """
    Convert all .tif files in the root_npz_dir to .jnpz files and save them in the root_npz_dir.
    """
    # if root_npz_dir exists, delete it and all contents, then create it
    # otherwise, just make it
    if os.path.exists(root_npz_dir):
        shutil.rmtree(root_npz_dir)
    os.makedirs(root_npz_dir)

    # use parse_img_mask_filenames and sort_filenames_in_series to get the correct order of the files
    # then convert the .tif files at those paths into numpy arrays
    for patient in tqdm(patients, unit="patient files converted from .tif to .npy"):
        patient_dir = os.path.join(root_tif_dir, patient)
        all_files = sorted(os.listdir(patient_dir))
        # sort the files into image and mask lists
        imgs, masks = parse_img_mask_filenames(all_files, '.tif')
        # sort the series into the correct order
        imgs = sort_filenames_in_series(imgs, patient, '.tif')
        masks = sort_filenames_in_series(masks, patient, '.tif')

        # if patient directory doesnt exist in PROCESSED_DATA_PATH, create it
        old_patient_dir = os.path.join(root_tif_dir, patient)
        new_patient_dir = os.path.join(root_npz_dir, patient)
        if not os.path.exists(new_patient_dir):
            os.makedirs(new_patient_dir)

        for img in imgs: img_npz_sample = convert_tif_to_npy(old_patient_dir, new_patient_dir, img, return_sample=True)
        for mask in masks: mask_npz_sample = convert_tif_to_npy(old_patient_dir, new_patient_dir, mask, return_sample=True)

    if return_sample:
        return img_npz_sample, mask_npz_sample
    else:
        return None, None


'''
.npy direcotry to .jpeg directory
'''
def npz_dirs_to_jpeg(root_npz_dir, root_jpg_dir, patients, return_sample=False):
    """
    Convert all .npy files in the root_npz_dir to .jpg files and save them in the root_jpg_dir.
    """
    # if root_jpg_dir exists, delete it and all contents, then create it
    # otherwise, just make it
    # if os.path.exists(root_jpg_dir):
    #     shutil.rmtree(root_jpg_dir)
    # os.makedirs(root_jpg_dir)

    for patient in tqdm(patients, unit="patient files converted from .npy to .jpg"):
        patient_dir = os.path.join(root_npz_dir, patient)
        all_files = sorted(os.listdir(patient_dir))
        # sort the files into image and mask lists
        imgs, masks = parse_img_mask_filenames(all_files, '.npy')
        # sort the series into the correct order
        imgs = sort_filenames_in_series(imgs, patient, '.npy')
        masks = sort_filenames_in_series(masks, patient, '.npy')
    
        # if patient directory doesnt exist in PROCESSED_DATA_PATH, create it
        old_patient_dir = os.path.join(root_npz_dir, patient)
        new_patient_dir = os.path.join(root_jpg_dir, patient)
        if not os.path.exists(new_patient_dir):
            os.makedirs(new_patient_dir)

        for img in imgs: img_jpg_sample = convert_npz_to_jpeg(old_patient_dir, new_patient_dir, img, return_sample=True)
        for mask in masks: mask_jpg_sample = convert_npz_to_jpeg(old_patient_dir, new_patient_dir, mask, return_sample=True)

    if return_sample:
        return img_jpg_sample, mask_jpg_sample
    else:
        return None, None
    
'''
.npy direcotry to .jpeg directory
'''
def npz_dirs_to_png(root_npz_dir, root_png_dir, patients, return_sample=False):
    """
    Convert all .npy files in the root_npz_dir to .jpg files and save them in the root_jpg_dir.
    """
    # if root_jpg_dir exists, delete it and all contents, then create it
    # otherwise, just make it
    # if os.path.exists(root_png_dir):
    #     shutil.rmtree(root_png_dir)
    # os.makedirs(root_png_dir)
    img_jpg_sample, mask_jpg_sample = None, None
    for patient in tqdm(patients, unit="patient files converted from .npy to .jpg"):
        patient_dir = os.path.join(root_npz_dir, patient)
        all_files = sorted(os.listdir(patient_dir))
        # sort the files into image and mask lists
        imgs, masks = parse_img_mask_filenames(all_files, '.npy')
        # sort the series into the correct order
        imgs = sort_filenames_in_series(imgs, patient, '.npy')
        masks = sort_filenames_in_series(masks, patient, '.npy')
    
        # if patient directory doesnt exist in PROCESSED_DATA_PATH, create it
        old_patient_dir = os.path.join(root_npz_dir, patient)
        new_patient_dir = os.path.join(root_png_dir, patient)
        if not os.path.exists(new_patient_dir):
            os.makedirs(new_patient_dir)
        for img in imgs: img_jpg_sample = convert_npz_to_png(old_patient_dir, new_patient_dir, img, return_sample=True)
        for mask in masks: mask_jpg_sample = convert_npz_to_png(old_patient_dir, new_patient_dir, mask, return_sample=True)

    if return_sample:
        return img_jpg_sample, mask_jpg_sample
    else:
        return None, None