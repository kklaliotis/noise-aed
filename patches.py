import numpy as np
from astropy.io import fits
import os
import re
from glob import glob
from scipy.ndimage import binary_dilation


def apply_object_mask(image, mask=None, threshold_factor=2.5, inplace=False):
    """
    Apply a bright object mask to an image.

    :param image: 2D numpy array, the image to be masked.
    :param mask: optional 2D boolean array, the pre-existing object mask.
    :param threshold_factor: float, threshold for masking a pixel
    :param: factor to multiply with the median for thresholding.
    :param inplace: whether to modify the input image directly.
    :return image_out: the masked image.
    :return neighbor_mask: the mask applied.
    """
    if mask is not None and isinstance(mask, np.ndarray):
        neighbor_mask = mask
    else:
        median_val = np.median(image)
        high_value_mask = image >= threshold_factor * median_val
        neighbor_mask = binary_dilation(high_value_mask, structure=np.ones((5, 5), dtype=bool))

    if inplace:
        image[neighbor_mask] = 0
        return image, neighbor_mask
    else:
        image_out = np.where(neighbor_mask, 0, image)
        return image_out, neighbor_mask

def extract_patches_from_image(image, scaid, patch_size=64, stride=32, normalize=False, science=False, mask_path=None):
    """
    Extract overlapping patches from a 2D image array.
    
    Parameters:
        image (2D np.ndarray): Input image, 4096x4224 pixels
        scaid (str): SCA ID, used for accessing the right mask.
        patch_size (int): Size of square patches.
        stride (int): Step between patch starts.
        normalize (bool): If True, normalize each patch to zero mean and unit std.
        science (bool): If True, apply object mask to the image.
        mask_path (str): Path to the permanent mask FITS file.

        
    Returns:
        patches (list of 2D np.ndarrays)
    """
    patches = []
    if image.shape == (4096, 4224):
        image = image[4:4092, 4:4092] # Crop to 4088x4088
    nrows, ncols = image.shape

    if mask_path:
        mask = fits.open(mask_path)[0].data[int(scaid) - 1].astype(bool)
        image *= mask
    else: raise ValueError("Permanent mask path must be provided.")

    if science:
        image = apply_object_mask(image, inplace=False)[0]

    for i in range(0, nrows - patch_size + 1, stride):
        for j in range(0, ncols - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size]
            if normalize:
                patch = patch - np.mean(patch)
                std = np.std(patch)
                if std > 0:
                    patch /= std
            patches.append(patch)
    return patches

def load_fits_images_and_extract_patches(dir, skip_obs=[670], patch_size=64, stride=32, max_files=None, science=False, mask_path=None):
    """
    Load FITS files from a directory and extract patches from each.
    
    Parameters:
        dir (str): Path to directory containing FITS files.
        patch_size (int): Patch size (default 64).
        stride (int): Patch stride (default 32).
        max_files (int or None): Limit number of files for testing.
        skip_obs (list): List of observation IDs to skip. 
        science (bool): If True, apply object mask to the images.
        mask_path (str): Path to the permanent mask FITS file.

    Returns:
        patches_array (np.ndarray): All extracted patches, shape (N, patch_size, patch_size)
    """
    all_patches = []

    if not science:
        nfiles=0
        files = sorted(glob(os.path.join(dir, "*.fits")))
        if max_files:
            files = files[:max_files]
        
        
        for filepath in files:
            m = re.match(r'.*?_(\d+)_([0-9]+)\.fits', filepath)
            if m:
                obsid=m.group(1)
                scaid=m.group(2)
        
            if int(obsid) not in skip_obs:
                nfiles+=1
                with fits.open(filepath) as hdul:
                    image = hdul['PRIMARY'].data.astype(np.float32)
                    patches = extract_patches_from_image(image, scaid, patch_size, stride, science=science, mask_path=mask_path)
                    all_patches.extend(patches)
            
        print(f"Extracted {len(all_patches)} patches from {nfiles} files in {dir}.")
        if not all_patches:
            raise ValueError("No patches extracted. Check the input directory and parameters.") 
        
    else: 
        m = re.match(r'.*?_(\d+)_([0-9]+)\.fits', dir)
        if m:
            scaid=m.group(2)
        with fits.open(dir) as hdul:
            image = hdul['SCI'].data.astype(np.float32)
            patches = extract_patches_from_image(image, scaid, patch_size, stride, science=science, mask_path=mask_path)
            all_patches.extend(patches)
    
    return np.array(all_patches)

def reconstruct_from_patches(patches, image_shape, patch_size=64, stride=32):
    """
    Reconstructs a full image from a set of overlapping patches (which go to one image)
    by averaging overlapping regions.
    
    Parameters:
    - patches: numpy array of shape (num_patches, patch_size, patch_size)
    - image_shape: tuple (H, W), the shape of the original image
    - patch_size: int, the size of the square patches
    - stride: int, the stride used during patch extraction

    Returns:
    - reconstructed_image: numpy array of shape image_shape
    """

    H, W = image_shape
    recon_image = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)

    patch_idx = 0
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            recon_image[y:y+patch_size, x:x+patch_size] += patches[patch_idx]
            weight_map[y:y+patch_size, x:x+patch_size] += 1
            patch_idx += 1

    # Handle any possible division by zero
    weight_map[weight_map == 0] = 1.0
    reconstructed = recon_image / weight_map
    return reconstructed


class Patches:
    """
    Class to handle patch extraction from FITS images.
    
    Attributes:
        dir (str): Directory containing FITS files.
        patch_size (int): Size of patches to extract.
        stride (int): Stride for patch extraction.
        max_files (int or None): Maximum number of files to process.
    """
    
    def __init__(self, dir, patch_size=128, skip_obs=[670], stride=None, max_files=None, science=False, mask_path=None):
        self.dir = dir
        self.patch_size = patch_size
        self.stride = patch_size//2 if stride is None else stride
        self.max_files = max_files
        self.science = science
        self.mask_path = mask_path
        self.skip_obs = skip_obs
        self.patches=None

    
    def extract(self):
        self.patches = load_fits_images_and_extract_patches(
            dir=self.dir, 
            patch_size=self.patch_size, 
            stride=self.stride, 
            max_files=self.max_files,
            science=self.science,
            mask_path=self.mask_path,
            skip_obs=self.skip_obs)

        return self.patches
    
    def stitch(self):
        assert self.patches is not None, "No patches extracted. Call extract() first."
        return reconstruct_from_patches(
            self.patches,
            image_shape=(4088, 4088),
            patch_size=self.patch_size,
            stride=self.stride)

    def patchify(self):
        return self.extract()
    
    def stitchify(self):
        return self.stitch()

