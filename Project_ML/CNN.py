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
#%%
#optics = Fluorescence(
#    NA=0.9,
#    magnification=15,
#    wavelength=510 * u.nm,
#    resolution=1.6 * u.um,
#    output_region=(0,0,128,128),
#)
optics = Fluorescence(
    NA=0.9,
    magnification=15,
    wavelength = 510*u.nm,
    resolution = 1.6 * u.um,
    output_region=(0,0,128,128),
)

#particle = PointParticle(
#    position=lambda:uniform(0,128, size=2),
#    intensity=lambda:uniform(3e6,5e6),
#    z=0,
#)

particle = PointParticle(
    position=lambda: uniform(0, 128, size=2),
    intensity=lambda: uniform(6e3,2e5),
    z=lambda: uniform(-0.5,0.5)*u.um,
)

#postprocess = (
#    Add(6e3) >> 
#    poisson
#)

normalization = (
    AsType("float") >> Subtract(110) >> Divide(250)
)

postprocess = (
    Add(lambda: uniform(100, 120)) >>
    poisson
)
#normalization = (
#    AsType(float) >> Subtract(6e3) >> Divide(2^32)
#)

simulation_pipeline = optics(particle) >> postprocess
simulated_image = simulation_pipeline()
print(simulated_image)
frame = 3
#platinum crop_width = 10 x = 30 - crop_width//2 y = 70 - crop_width//2
crop_width = 15
x = 82- crop_width//2
y = 51 - crop_width//2
image = io.imread(image_paths[frame])
crop = image[x:x+crop_width, y:y+crop_width]
simulated_image = gaussian_filter(simulated_image,1.3)
simulated_image = gaussian_filter(simulated_image,1.3)
plt.figure(figsize=(10,10))

plt.subplot(1,2,1)
plt.imshow(crop, cmap="gray", vmin=0, vmax=30000)
plt.axis("off")
plt.title("Experimental particle")

plt.subplot(1,2,2)
plt.imshow(simulated_image, cmap="gray",vmin=0,vmax=30000)
plt.axis("off")
plt.title("Simulated particle")

plt.tight_layout()
plt.show()

#%%
sample = particle^10
simulation_pipeline = optics(sample) >> postprocess >> normalization
image = simulation_pipeline()
plt.figure(figsize=(10,10))
plt.imshow(image, cmap="gray")
plt.axis("off")
plt.show()
#%%
frame = 3
#platinum crop_width = 10 x = 30 - crop_width//2 y = 70 - crop_width//2
crop_width = 14
x = 83- crop_width//2
y = 52 - crop_width//2
image = io.imread(image_paths[frame])
crop = image[x:x+crop_width, y:y+crop_width]
optics = Fluorescence(
    NA=0.9,
    magnification=15,
    wavelength = 510*u.nm,
    resolution = 1.6 * u.um,
    output_region=(0,0,14,14),
)
particle = PointParticle(
    position=(7,7),
    intensity=6e6,
    z=0,
)

postprocess = (
    Add(6e3)
)

noise = Poisson(
    min_snr = 100,
    max_snr = 1000,
    snr= lambda min_snr, max_snr: min_snr + np.random.rand() * (max_snr - min_snr),
    background = 0
)

noise2 = Gaussian(
    sigma=5
)


simulation_pipeline = optics(particle) >> postprocess >> noise >> noise2
simulated_image = simulation_pipeline()

plt.figure(figsize=(10,10))

plt.subplot(1,2,1)
plt.imshow(crop, cmap="gray", vmin=0, vmax=35000)
plt.axis("off")
plt.title("Experimental particle")

plt.subplot(1,2,2)
plt.imshow(simulated_image, cmap="gray", vmin=0, vmax=35000)
plt.axis("off")
plt.title("Simulated particle")

plt.tight_layout()
plt.show()

optics = Fluorescence(
    NA=0.9,
    magnification=20,
    wavelength = 510*u.nm,
    resolution = 1.6 * u.um,
    output_region=(0,0,128,128)
)
particle = PointParticle(
    position=lambda: uniform(0, 128, size=2),
    intensity=lambda: uniform(5e6,1e7),
    z=lambda: uniform(-0.0,0.0)*u.um
)

postprocess = (
    Add(lambda: uniform(0, 0)) >>
    poisson
)

normalization = (
    AsType("float")
)

sample = particle^20
number_of_particles = 100
simulated_image = np.zeros([128,128])
for i in range(number_of_particles):
    x = i * 64/10 % 128
    y = (i% 128/10)*10
    print(x, y)
    simulation_pipeline = optics(PointParticle(position = [x,y],
                                                intensity = uniform(5e6,1.3e7),
                                                z=0)
                                ) >> noise >> noise2
    print(np.shape(np.squeeze(np.array(simulation_pipeline()), axis=2)))
    simulated_image = simulated_image + np.squeeze(np.array(simulation_pipeline()), axis=2)
#simulated_image = gaussian_filter(simulated_image,0.01)


#simulation_pipeline = optics(sample) >> postprocess >> normalization
#image = simulation_pipeline()
#simulated_image = gaussian_filter(image,1.3)
#simulated_image = gaussian_filter(simulated_image,1.3)
plt.figure(figsize=(10,10))
plt.imshow(simulated_image, cmap="gray")
plt.axis("off")
plt.show()
plt.hist(np.array(simulated_image).ravel(),bins=255)
print(max(np.array(simulated_image).ravel()))
# %%
