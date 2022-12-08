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

dataset_path = "data/experimental_data/32bit"
image_paths = sorted(glob.glob(os.path.join(dataset_path,"*.tif")))

optics_crop = Fluorescence(
    NA=0.9,
    magnification=50,
    wavelength = 510*u.nm,
    resolution = 1.6 * u.um,
    output_region=(0,0,15,15),
)

optics_image = Fluorescence(
    NA=0.9,
    magnification= 27,
    wavelength = 510*u.nm,
    resolution = 1.6 * u.um,
    output_region=(0,0,128,128),
)

optics_image_2 = Fluorescence(
    NA=0.9,
    magnification= 35,
    wavelength = 510*u.nm,
    resolution = 1.6 * u.um,
    output_region=(0,0,128,128),
)

optics_noise = Fluorescence(
    NA=0.9,
    magnification=40,
    wavelength = 510*u.nm,
    resolution = 1.6 * u.um,
    output_region=(0,0,128,128),
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
    Add(lambda: uniform(0,3000))
)

noise_poisson = Poisson(
    min_snr = 3000,
    max_snr = 7000,
    snr= lambda min_snr, max_snr: min_snr + np.random.rand() * (max_snr - min_snr),
    background = 0
)

noise_gaussian = Gaussian(
    sigma=100
)


def generate_structure(image_size,number_of_particles,border):
    temp = (image_size-2*border)/math.sqrt(number_of_particles)
    label_image = np.zeros((image_size,image_size))
    assert number_of_particles < math.pow(math.floor(temp),2)
    positions = np.zeros((number_of_particles,2))
    frame = np.zeros((image_size,image_size))
    frame[:,image_size-border:image_size] = 1
    frame[:,0:border] = 1
    frame[0:border,:] = 1
    frame[image_size-border:image_size,:] = 1
    for i in range(number_of_particles):
        idx = randint(0,np.shape(np.where(frame == 0))[1])
        x_int,y_int = [np.where(frame == 0)[0][idx],np.where(frame == 0)[1][idx]]
        x,y = [uniform(x_int-0.499,x_int+0.499),uniform(y_int-0.499,y_int+0.499)]
        positions[i] = [x,y]
        frame[x_int-border:x_int+border,y_int-border:y_int+border] = 1

    for idx,pos in enumerate(positions):
        x,y = pos
        x_int,y_int= np.floor(pos)
        x_int = int(x_int)
        y_int = int(y_int)
        gt_neighborhood = [[x_int-1,y_int-1],[x_int,y_int-1],[x_int+1,y_int-1],
                        [x_int-1,y_int],[x_int,y_int],[x_int+1,y_int],
                        [x_int-1,y_int+1],[x_int,y_int+1],[x_int+1,y_int+1]]
        for idx,pixel in enumerate(gt_neighborhood):
            #print(gt_neighborhood,pos)
            pixel_center = np.array(pixel)+0.5
            label_image[pixel[1],pixel[0]] = 1/(math.pow(math.e,math.dist(pixel_center,pos)))
    return positions, label_image
        
image_size=128
positions, labels = generate_structure(128,70,border=8)  
simulated_image = np.zeros((image_size,image_size))
noise_particles = randint(0,10)
for i in range(noise_particles):
    y = uniform(0,128)
    x = uniform(0,128)
    simulation_pipeline = optics_noise(PointParticle(position = [x,y],
                                                intensity = uniform(7e9,9e9),
                                                z=lambda: uniform(25,40))
                                    )
    simulated_image = simulated_image + np.squeeze(np.array(simulation_pipeline()), axis=2)
for i in range(np.shape(positions)[0]):
    y = positions[i][0]
    x = positions[i][1]
    if randint(0,2):
        simulation_pipeline = optics_image(PointParticle(position = [x,y],
                                                    intensity = uniform(7e8,1.5e9),
                                                    z=0)
                                    ) >> noise_poisson
    else:
        simulation_pipeline = optics_image_2(PointParticle(position = [x,y],
                                                    intensity = uniform(7e8,3e9),
                                                    z=0)
                                    ) >> noise_poisson        
    simulated_image = simulated_image + np.squeeze(np.array(simulation_pipeline()), axis=2)
kernel = np.ones((3,3))
number_of_particles = np.shape(positions)[0]
simulated_image = gaussian_filter(simulated_image/number_of_particles,uniform(0.6,0.8))


#simulation_pipeline = optics(sample) >> postprocess >> normalization
#image = simulation_pipeline()
#simulated_image = gaussian_filter(image,1.3)
#simulated_image = gaussian_filter(simulated_image,1.3)
plt.figure(figsize=(10,10))
plt.imshow(simulated_image, cmap="gray")
plt.scatter(positions[:,0],positions[:,1])
plt.axis("off")
plt.show()
plt.hist(np.array(simulated_image).ravel(),bins=255)
print(max(np.array(simulated_image).ravel()))
#%%

#Generate_dataset
number_of_images = 1000
image_size = 128
#number_of_particles = 70
save_dir = "data/training_data"

image_cube=np.zeros((image_size,image_size,number_of_images))
label_cube=np.zeros((image_size,image_size,number_of_images))
for i in range(number_of_images):
    number_of_particles = randint(60,80)
    simulated_image = np.zeros((image_size,image_size))
    positions, label_image = generate_structure(image_size,number_of_particles,border=8)
    noise_particles = randint(0,10)
    for k in range(noise_particles):
        y = uniform(0,128)
        x = uniform(0,128)
        simulation_pipeline = optics_noise(PointParticle(position = [x,y],
                                                intensity = uniform(7e9,9e9),
                                                z=lambda: uniform(25,40))
                                        ) >> normalization
        simulated_image = simulated_image + np.squeeze(np.array(simulation_pipeline()), axis=2)
    for j in range(np.shape(positions)[0]):
        y = positions[j][0]
        x = positions[j][1]
        if randint(0,2):
            simulation_pipeline = optics_image(PointParticle(position = [x,y],
                                                        intensity = uniform(7e8,1.5e9),
                                                        z=0)
                                        ) >> noise_poisson >> noise_gaussian >> normalization
        else:
            simulation_pipeline = optics_image_2(PointParticle(position = [x,y],
                                                        intensity = uniform(7e8,3e9),
                                                        z=0)
                                        ) >> noise_poisson >> noise_gaussian >> normalization
        simulated_image = simulated_image + np.squeeze(np.array(simulation_pipeline()), axis=2)            
    kernel = np.ones((3,3))
    number_of_particles = np.shape(positions)[0]
    simulated_image = gaussian_filter(simulated_image/number_of_particles,uniform(0.6,0.8))
    image_cube[:,:,i] = simulated_image
    label_cube[:,:,i] = label_image
    #print("image ",min(np.array(image_cube[:,:,i]).ravel()),"--",max(np.array(image_cube[:,:,i]).ravel()))
    #print(max(np.array(label_image).ravel()),min(np.array(label_image).ravel()))
    #print(min(np.array(simulated_image).ravel()),max(np.array(simulated_image).ravel()))
    #np.save(save_dir+f'/{i}_Label',simulated_image)
    print(f'{i+1}/{number_of_images}')
#%%
image_cube = image_cube - min(np.array(image_cube[:,:,:]).ravel())
np.save(save_dir+'/Labels',label_cube)
np.save(save_dir+'/Images',image_cube)
# %%
d = np.load(save_dir+'/Images.npy')
#assert d == image_cube
print(d == image_cube)
print("image ",min(np.array(image_cube[:,:,0]).ravel()),"--",max(np.array(image_cube[:,:,0]).ravel()))
image = image_cube[:,:,0]
label = label_cube[:,:,0]
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.imshow(image,cmap="gray")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(label,cmap="gray")
plt.axis("off")
    #plt.tight_layout()

# %%
idx=random.sample(range(number_of_images), 5)
print(idx)
plt.figure(figsize=(15,15))
for idx, val in enumerate(idx):
    idx=idx*2
    image = image_cube[:,:,val]
    label = label_cube[:,:,val]
    plt.subplot(5,2,idx+1)
    plt.imshow(image,cmap="gray")
    plt.axis("off")
    plt.subplot(5,2,idx+2)
    plt.imshow(label,cmap="gray")
    plt.axis("off")
    plt.tight_layout()
# %%
