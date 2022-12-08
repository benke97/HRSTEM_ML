#%%
from deeptrack import Fluorescence, Poisson, Gaussian, PointParticle, Add, Subtract, Divide, AsType, units as u
from numpy.random import poisson, uniform
import matplotlib.pyplot as plt
import os
import glob
import skimage
from skimage import io
from scipy.ndimage import gaussian_filter
import numpy as np

dataset_path = "data/experimental_data/32bit"
image_paths = sorted(glob.glob(os.path.join(dataset_path,"*.tif")))

optics = Fluorescence(
    NA=0.9,
    magnification=30,
    wavelength = 510*u.nm,
    resolution = 1.6 * u.um,
    output_region=(0,0,15,15),
)

particle = PointParticle(
    position=[7.5,7.5],
    intensity=3e7,
    z=lambda: uniform(-0.5,0.5)*u.um,
)

normalization = (
    Divide(4294967296)
)

postprocess = (
    Add(5000)
)

noise_poisson = Poisson(
    min_snr = 500,
    max_snr = 1000,
    snr= lambda min_snr, max_snr: min_snr + np.random.rand() * (max_snr - min_snr),
    background = 0
)

noise_gaussian = Gaussian(
    sigma=1
)

simulation_pipeline =optics(particle) >>postprocess >> noise_poisson >> normalization
simulated_image = gaussian_filter(simulation_pipeline(),1)
#simulated_image = simulation_pipeline()
frame = 3

crop_width = 15
x = 82- crop_width//2
y = 51 - crop_width//2
image = io.imread(image_paths[frame])
crop = image[x:x+crop_width, y:y+crop_width]
plt.figure(figsize=(10,10))

plt.subplot(1,2,1)
plt.imshow(crop, cmap="gray", vmin=0, vmax=2*max(np.array(crop).ravel()))
plt.axis("off")
plt.title("Experimental particle")

plt.subplot(1,2,2)
plt.imshow(simulated_image, cmap="gray",vmin=0,vmax=2*max(np.array(crop).ravel()))
plt.axis("off")
plt.title("Simulated particle")

plt.tight_layout()
plt.show()
print(max(np.array(crop).ravel()))
print(max(np.array(simulated_image).ravel()))


#%%
plt.figure(figsize=(10,10))
plt.subplot(10,10,1)
plt.imshow(crop, cmap="gray", vmin=0, vmax=2*max(np.array(crop).ravel()))
plt.axis("off")
plt.title("Experimental particle")
for i in range(99):
    simulation_pipeline =optics(particle) >>postprocess >> noise_poisson
    simulated_image = gaussian_filter(simulation_pipeline(),1)
    plt.subplot(10,10,i+2)
    plt.imshow(simulated_image, cmap="gray",vmin=0,vmax=2*max(np.array(crop).ravel()))
    plt.axis("off")
# %%
