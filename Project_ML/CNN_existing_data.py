#%%
from deeptrack.models import UNet
from deeptrack.losses import weighted_crossentropy
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from IPython.display import clear_output
from skimage import io
import random

random.seed(1337)

# %%

dataset_path = 'data/training_data/'
images_path = dataset_path+"Images.npy"
labels_path = dataset_path+'Labels.npy'
images = np.load(images_path)*100000
labels = np.load(labels_path)

print(np.shape(images[...,np.newaxis]))
new_images = np.swapaxes(images,0,2)
new_labels = np.swapaxes(labels,0,2)[...,np.newaxis]
print(np.shape(new_images[...,np.newaxis]))
print(np.shape(new_labels))
print("label ",min(np.array(labels[:,:,:]).ravel()),"--",max(np.array(labels[:,:,:]).ravel()))
print("image ",min(np.array(images[:,:,:]).ravel()),"--",max(np.array(images[:,:,:]).ravel()))
print(np.dtype(labels[1,1,1]))
print(np.dtype(images[1,1,1]))

#%%
model = UNet(
    input_shape=(128,128,1),
    conv_layers_dimensions=(16,32),
    base_conv_layers_dimensions=(64,64),
    output_activation="sigmoid"
)

model.summary()

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=12*1000,
  decay_rate=1,
  staircase=False)

def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)


model.compile(
    loss=weighted_crossentropy((500,1)),
    optimizer="adam"
    #optimizer = tf.keras.optimizers.Adam(lr_schedule)
    #optimizer="tf.keras.optimizers.Adam(learning_rate=1e-3)"
)

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(display_list[i],cmap="gray")
        plt.axis('off')
    plt.show()

def show_predictions():
    predicted_mask = model.predict(new_images[1,:,:][tf.newaxis, ...])
    display([new_images[1,:,:], new_labels[1,:,:], np.squeeze(predicted_mask,axis=0)])

show_predictions()

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

#%%
model.fit(
    new_images[0:900,:,:],
    new_labels[0:900,:,:],
    epochs=50,
    batch_size=32,
    callbacks=[DisplayCallback()],
)
# %%
image_of_particle = np.squeeze(new_images[901:1000,:,:])[1,:,:]
gt_mask = np.squeeze(new_labels[901:1000,:,:])[1,:,:]
plt.figure()
plt.subplot(1,2,1)
plt.imshow(image_of_particle,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(gt_mask,cmap="gray")

predicted_mask = model.predict_on_batch(image_of_particle[np.newaxis,...])

plt.figure()
plt.subplot(1,2,1)
plt.imshow(gt_mask,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(np.squeeze(predicted_mask),cmap="gray")
print("gt label ",min(np.array(gt_mask).ravel()),"--",max(np.array(gt_mask).ravel()))
print("predicted label ",min(np.array(predicted_mask).ravel()),"--",max(np.array(predicted_mask).ravel()))
# %% Experimental data
dataset_path = "data/experimental_data/32bit"
image_paths = sorted(glob.glob(os.path.join(dataset_path,"*.tif")))
print(image_paths)
image = io.imread(image_paths[2])
plt.figure()
plt.subplot(1,2,1)
plt.imshow(image,cmap="gray")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image_of_particle,cmap="gray")
plt.axis("off")
image = image*100000/4294967296
print("",min(np.array(image).ravel()),"--",max(np.array(image).ravel()))

predicted_mask2 = model.predict_on_batch(image[np.newaxis,...])
plt.figure()
plt.subplot(1,2,1)
plt.imshow(image,cmap="gray")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(np.squeeze(predicted_mask2),cmap="gray")
plt.axis("off")
print(len(image_paths))
# %%
for idx,path in enumerate(image_paths):
    plt.figure(figsize=(15,15))
    idx = idx*2
    image = io.imread(path)
    image = image*100000/4294967296
    predicted_mask = model.predict_on_batch(image[np.newaxis,...])
    plt.subplot(1,2,1)
    plt.imshow(image,cmap="gray")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(np.squeeze(predicted_mask),cmap="gray")
    plt.axis("off")
# %%
