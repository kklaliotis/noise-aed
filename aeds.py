import numpy as np
from astropy.io import fits
import os
import re
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from patches import Patches

class ConvAutoencoder(nn.Module):
    """
    A simple convolutional autoencoder for image denoising.
    It consists of an encoder that compresses the input image into a latent representation,
    and a decoder that reconstructs the image from this representation.
    """

    def __init__(self):
        super().__init__()

        # Encoder : 1x64x64  -> 64x8x8
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class PatchDataset(Dataset):
    """ A PyTorch Dataset for loading patches.
    It assumes that the patches are stored in a numpy array format.
    """

    def __init__(self, patch_array):
        self.patches = patch_array  # Shape: (N, 64, 64)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        return torch.tensor(patch, dtype=torch.float32).unsqueeze(0)  # (1, 64, 64)

# Hard code for now -- make configurable later
darks_path = "/fs/scratch/PCON0003/cond0007/anl-run-in-prod/labnoise"
science_path = "/fs/scratch/PCON0003/klaliotis/noise-aed/inputs"
pmask_path = "/users/PCON0003/cond0007/imcom/coadd-test-fall2022/permanent_mask_220730.fits"
skip_obs = [670]
patch_size = 128
stride = 64
max_files = None

print("Extracting noise patches from darks...")

noise_patcher = Patches(
    dir=darks_path,
    patch_size=patch_size,
    skip_obs=skip_obs,
    stride=stride,
    mask_path=pmask_path,
    max_files=max_files,
    science=False
)

noise_patches = noise_patcher.patchify()

print(f"Extracted {len(noise_patches)} noise patches of size {patch_size}x{patch_size}.")

# Create datasets
train_dataset = PatchDataset(noise_patches)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model
model = ConvAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Starting training...")

# Training loop
for epoch in range(20):
    model.train()
    loss_total = 0
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
    print(f"Epoch {epoch}: loss = {loss_total:.4f}")

print("Training complete. Saving model to aeds_model.pt")

torch.save(model.state_dict(), 'aeds_model.pt')

# for the nexts ection i want to go through science images one at a time. maybe i can do an input dir and then make alist of the files in that dir 
# or input dir+prefix->obsid and then make a list of those and then do this thing on a loop for each file. nbut then i have to adapt
# the patches thing to be bale to use just one image for patching science images

print(" Beginning patch extraction from science images...")

files = sorted(glob(os.path.join(science_path, "*.fits")))
if max_files:
    files = files[:max_files]

for file in files:
    m = re.match(r'.*?_(\d+)_([0-9]+)\.fits', file)
    if m:
        obsid = m.group(1)
        scaid = m.group(2)

        if int(obsid) in skip_obs:
            print(f"Testing model on file {file}")

            science_patcher = Patches(
                dir=file,
                patch_size=patch_size,
                skip_obs=skip_obs,
                stride=stride,
                mask_path=pmask_path,
                max_files=max_files,
                science=True
            )

            science_patches = science_patcher.patchify()

            print(f"Extracted {len(science_patches)} science patches of size {patch_size}x{patch_size}.")

            model.eval()
            with torch.no_grad():
                reconstructed_patches = []
                for patch in science_patches:
                    patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  
                    rec = model(patch_tensor).squeeze().numpy() 
                    reconstructed_patches.append(rec)

            reconstructed_patches = np.array(reconstructed_patches)

            print(f"Reconstructed {len(reconstructed_patches)} patches from science images.")

            science_patcher.patches = reconstructed_patches
            reconstructed_image = science_patcher.stitchify()
            print(f"Reconstructed image shape: {reconstructed_image.shape}")

            original_image = np.copy(fits.open(file)['SCI'].data.astype(np.float32))
            print(f"Original image shape: {original_image.shape}")

            # Save the reconstructed image
            output_file = f"{obsid}_{scaid}_aeds.fits"
            hdu = fits.PrimaryHDU(original_image-reconstructed_image)
            hdu2 = fits.ImageHDU(reconstructed_image)
            hdul = fits.HDUList([hdu, hdu2])
            hdul.writeto(output_file, overwrite=True)
            print(f"Reconstructed image saved to {output_file}")

print("All science images processed. Complete!!")

