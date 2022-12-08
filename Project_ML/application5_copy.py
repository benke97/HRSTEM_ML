#%%
from deeptrack import Fluorescence, Poisson, Gaussian, SampleToMasks, PointParticle, Add, Subtract, Divide, AsType, units as u
from numpy.random import poisson, uniform, randint
import matplotlib.pyplot as plt
import os
import glob
import skimage
from skimage import io
from scipy.ndimage import gaussian_filter
import numpy as np
import cv2
import math
import random

#%%
my_optics_image = Fluorescence(
    NA=0.9,
    magnification=28,
    wavelength = 510*u.nm,
    resolution = 1.6 * u.um,
    output_region=(0,0,128,128),
)

my_particle=PointParticle(position = lambda: uniform(0,128,size=2),
                                                intensity = lambda: uniform(1e7,2e7),
                                                z=0)

my_normalization = (
    AsType('float') >> Divide(4294967296)
)

my_postprocess = (
    poisson
)

noise_poisson = Poisson(
    min_snr = 4000,
    max_snr = 4000,
    snr= lambda min_snr, max_snr: min_snr + np.random.rand() * (max_snr - min_snr),
    background = 0
)

noise_gaussian = Gaussian(
    sigma=200
)


optics_image = Fluorescence(
    NA=0.9,
    magnification=20,
    wavelength = 510*u.nm,
    resolution = 1.6 * u.um,
    output_region=(0,0,128,128),
)

particle=PointParticle(position = lambda: uniform(0,128,size=2),
                                                intensity = lambda: uniform(6e3,2e5),
                                                z=0)

postprocess = (
    Add(lambda: uniform(100,120)) >>
    poisson
)

normalization = (
    AsType("float") >> Subtract(110) >> Divide(250)
)
sample = particle^70
simulation_pipeline = optics_image(sample) >> postprocess >> normalization
simulated_image = simulation_pipeline()

my_sample = my_particle^70
my_simulation_pipeline = my_optics_image(my_sample) >> noise_poisson >>noise_gaussian >> my_normalization
my_simulated_image = my_simulation_pipeline()
plt.figure(figsize=(10,10))
plt.imshow(simulated_image, cmap="gray")
plt.axis("off")
plt.show()
plt.hist(np.array(simulated_image).ravel(),bins=255)
print(max(np.array(simulated_image).ravel()))
# %%

gt_masks = sample >> SampleToMasks(
    lambda: lambda particle: particle > 0,
    output_region=my_optics_image.output_region,
    merge_method="or"
)
image_and_mask_pipeline = simulation_pipeline & gt_masks
image, gt_mask = image_and_mask_pipeline()


my_gt_masks = my_sample >> SampleToMasks(
    lambda: lambda particle: particle > 0,
    output_region=my_optics_image.output_region,
    merge_method="or"
)
my_image_and_mask_pipeline = my_simulation_pipeline & my_gt_masks
my_image, my_gt_mask = my_image_and_mask_pipeline()

print("label ",min(np.array(gt_mask).ravel()),"--",max(np.array(gt_mask[:,:,:]).ravel()))
print("image ",min(np.array(image[:,:,:]).ravel()),"--",max(np.array(image[:,:,:]).ravel()))
print(np.dtype(gt_mask))


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(gt_mask, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
# %%
from deeptrack.models import UNet

model = UNet(
    input_shape=(None,None,1),
    conv_layers_dimensions=(16,32),
    base_conv_layers_dimensions=(64,64),
    output_activation="sigmoid"
)
model.summary()

# %%
from deeptrack.losses import weighted_crossentropy

model.compile(
    loss=weighted_crossentropy((1000,1)),
    optimizer="adam"
)

model.fit(
    image_and_mask_pipeline,
    epochs=50,
    batch_size=32
)
# %%
image_and_mask_pipeline.update()
image_of_particle, gt_mask = image_and_mask_pipeline()
plt.figure()
plt.subplot(1,2,1)
plt.imshow(image_of_particle)
plt.subplot(1,2,2)
plt.imshow(gt_mask)

predicted_mask = model.predict_on_batch(image_of_particle[np.newaxis])

plt.figure()
plt.subplot(1,2,1)
plt.imshow(gt_mask)
plt.subplot(1,2,2)
plt.imshow(np.squeeze(predicted_mask))
# %%
