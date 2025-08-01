import re
import os
from astropy.io import fits
import numpy as np


"""
Sums the FITS image at sci_path with a matching image in noise_dir,
using the 'number_number' pattern in the filename, and writes the result to out_dir.

Parameters:
    sci_path (str): Path to the input FITS file (e.g., science image).
    noise_dir (str): Directory to search for the matching FITS file (e.g., noise image).
    out_dir (str): Directory where the output summed image will be saved.
"""

sci_path_prefix = '/fs/scratch/PCON0003/cond0007/anl-run-in-prod/simple/Roman_WAS_simple_model_H158_670_'
noise_dir = '/fs/scratch/PCON0003/cond0007/anl-run-in-prod/labnoise'
out_dir = '/fs/scratch/PCON0003/klaliotis/noise-aed/inputs'
obs_include = [670, 13908]

for obs in obs_include:
    obsid=str(obs)

    for i in range(1,19):
        sci_filename = sci_path_prefix + str(i) + '.fits'
        basename = os.path.basename(sci_filename)

        match = re.search(fr'({obsid})_(1[0-9]|[1-9])\.fits$', basename)
        if not match:
            raise ValueError(f"No 'number_number' pattern found in filename: {basename}")
        pattern = f"{match.group(1)}_{match.group(2)}"

        # Search for matching file in noise_dir
        matched_file = None
        for fname in os.listdir(noise_dir):
            if pattern in fname and fname.endswith('.fits'):
                matched_file = os.path.join(noise_dir, fname)
                break

        print(f"Matched file: {matched_file} for pattern {pattern} in basename {basename}")

        if matched_file is None:
            print(f"Warning: No matching file found in {noise_dir} for pattern {pattern}. Skipping.")
            continue  # skip to the next iteration

        with fits.open(sci_filename) as hdul1, fits.open(matched_file) as hdul2:
            sci = hdul1['SCI'].data.astype(np.float32)
            noise = hdul2['PRIMARY'].data.astype(np.float32)[4:4092, 4:4092] * 1.458 * 50  # gain * N_frames
            header = hdul1[0].header

        # Sum the images
        summed = sci + noise

        # Create HDUs
        primary_hdu = fits.PrimaryHDU(summed, header=header)
        sci_hdu = fits.ImageHDU(sci, name='SCI')
        noise_hdu = fits.ImageHDU(noise, name='NOISE')

        # Combine into HDUList
        hdulist = fits.HDUList([primary_hdu, sci_hdu, noise_hdu])

        # Write out the file
        out_name = f"Roman_WAS_Noise_H158_{pattern}.fits"
        out_path = os.path.join(out_dir, out_name)
        hdulist.writeto(out_path, overwrite=True)

        print(f"Saved summed image to {out_path}")


