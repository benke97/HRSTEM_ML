#%%
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PI_U_Net import UNet
import time
import pickle

def preprocess_image(image_path):
    raw_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    normalized_image = np.maximum((raw_image - raw_image.min()) / (raw_image.max() - raw_image.min()), 0)
    normalized_image = normalized_image[np.newaxis, :, :]
    image_tensor = torch.tensor(normalized_image, dtype=torch.float32)
    return image_tensor

def postprocess_output(output_tensor):
    output_numpy = output_tensor.detach().cpu().numpy()
    output_image = np.squeeze(output_numpy)
    return output_image
#%%
model = UNet()

loaded_data = torch.load("model_data_epoch_44.pth")

# Extract the epoch number, model state dictionary, and validation loss
loaded_epoch = loaded_data['epoch']
loaded_model_state_dict = loaded_data['model_state_dict']
#loaded_validation_loss = loaded_data['validation_loss']
model.load_state_dict(loaded_model_state_dict)
image_path = "data/experimental_data/32bit/01.tif"

start_time = time.time()
image_tensor = preprocess_image(image_path)
image_tensor = image_tensor.unsqueeze(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_tensor = image_tensor.to(device)
model = model.to(device)
# Pass the image tensor through the model
model.eval()
with torch.no_grad():
    predicted_output = model(image_tensor)

predicted_segmentation = postprocess_output(predicted_output)
input_image_np = image_tensor.squeeze(0).squeeze(0).cpu().numpy()
end_time = time.time()
time_taken = end_time - start_time
print(time_taken)
# Plot the input image and predicted segmentation
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(input_image_np, cmap='gray')
plt.title(f'Epoch{loaded_epoch}')

plt.subplot(1, 2, 2)
plt.imshow(np.where((predicted_segmentation > 0.01),predicted_segmentation,predicted_segmentation), cmap='gray')
plt.title('Predicted Segmentation')

plt.show()
# %%

with open('large_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = time.time()
outputs = []
for idx,image in enumerate(data['images']):
    normalized_image = np.maximum((image - image.min()) / (image.max() - image.min()), 0)
    normalized_image = normalized_image[np.newaxis, :, :]
    image_tensor = torch.tensor(normalized_image, dtype=torch.float32)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device) 
    model.eval()
    with torch.no_grad():
        predicted_output = model(image_tensor)
        predicted_segmentation = postprocess_output(predicted_output)
        outputs.append(predicted_segmentation)
end_time = time.time()
time_taken = end_time - start_time
print(time_taken)
print(len(outputs))
# %%
from Analyzer import Analyzer
inputs = data['images']
start_time = time.time()
analyzer = Analyzer(inputs,outputs,data["dataframes"])
analyzer.calculate_average_error()
end_time = time.time()
time_taken = end_time - start_time
print(time_taken)

#%%
from Analyzer import Analyzer
import cupy as cp

with open('large_dataset_with_predictions.pkl', 'rb') as f:
    data = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet()
model = model.to(device)
means = []
stds = []
start_epoch = 25
end_epoch = 50
best_epoch = 0
min_mean = float('inf')
for epoch in range(start_epoch, end_epoch + 1):
    start_time = time.time()
    print(epoch)
    loaded_data = torch.load(f"model_data_epoch_{epoch}.pth")

    # Extract the epoch number, model state dictionary, and validation loss
    loaded_epoch = loaded_data['epoch']
    loaded_model_state_dict = loaded_data['model_state_dict']
    # loaded_validation_loss = loaded_data['validation_loss']
    model.load_state_dict(loaded_model_state_dict)
    outputs = []
    for idx, image in enumerate(data['images']):
        image_cp = cp.array(image)
        normalized_image = cp.maximum((image_cp - image_cp.min()) / (image_cp.max() - image_cp.min()), 0)
        normalized_image_np = cp.asnumpy(normalized_image)  # Convert back to numpy for torch.tensor()
        normalized_image_np = normalized_image_np[np.newaxis, :, :]
        image_tensor = torch.tensor(normalized_image_np, dtype=torch.float32)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)
        model.eval()
        with torch.no_grad():
            predicted_output = model(image_tensor)
            predicted_segmentation = postprocess_output(predicted_output)
            outputs.append(predicted_segmentation)

    inputs = data['images']
    analyzer = Analyzer()
    analyzer.set_image_list(inputs)
    analyzer.set_gt_list(data['exponential_ground_truths'])
    analyzer.set_pred_list(outputs)
    analyzer.set_point_sets(data["dataframes"])
    
    mean, std = analyzer.calculate_average_error()
    means.append(mean)
    stds.append(std)
    if mean < min_mean:
        min_mean = mean
        best_epoch = epoch
    end_time = time.time()
    time_taken = end_time - start_time
    print(time_taken,min_mean)

# Create a common x-axis for both lists (assuming equal length)
x = cp.arange(start_epoch, end_epoch + 1)

# Create the main plot with the left y-axis
fig, ax1 = plt.subplots()
ax1.plot(x.get(), means, 'b-')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Mean Error', color='b')
ax1.tick_params('y', colors='b')

# Add 'x' symbol at the best_epoch
ax1.plot(best_epoch, min_mean, 'x', color='g', markersize=10)

# Create a label with the epoch number
best_epoch_label = f"Best Epoch: {best_epoch}"
ax1.legend([best_epoch_label], loc='upper left')

# Create a secondary y-axis for the second list
ax2 = ax1.twinx()
ax2.plot(x.get(), stds, 'r-')
ax2.set_ylabel('Std', color='r')
ax2.tick_params('y', colors='r')

# Add a title and display the plot
plt.title('Error vs Epoch')
plt.show()

# %%
from Analyzer import Analyzer
import cupy as cp
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('large_dataset_with_predictions.pkl', 'rb') as f:
    data = pickle.load(f)
model = UNet()
model = model.to(device)
loaded_data = torch.load("model_data_epoch_45.pth")

# Extract the epoch number, model state dictionary, and validation loss
loaded_epoch = loaded_data['epoch']
loaded_model_state_dict = loaded_data['model_state_dict']
#loaded_validation_loss = loaded_data['validation_loss']
model.load_state_dict(loaded_model_state_dict)
image_path = "data/experimental_data/32bit/09.tif"

outputs = []
for idx, image in enumerate(data['images']):
    image_cp = cp.array(image)
    normalized_image = cp.maximum((image_cp - image_cp.min()) / (image_cp.max() - image_cp.min()), 0)
    normalized_image_np = cp.asnumpy(normalized_image)  # Convert back to numpy for torch.tensor()
    normalized_image_np = normalized_image_np[np.newaxis, :, :]
    image_tensor = torch.tensor(normalized_image_np, dtype=torch.float32)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    model.eval()
    with torch.no_grad():
        predicted_output = model(image_tensor)
        predicted_segmentation = postprocess_output(predicted_output)
        outputs.append(predicted_segmentation)
inputs = data['images']
analyzer = Analyzer(inputs, data['exponential_ground_truths'], outputs, data["dataframes"])
image_tensor = preprocess_image(image_path)
image_tensor = image_tensor.unsqueeze(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_tensor = image_tensor.to(device)
model = model.to(device)
# Pass the image tensor through the model
model.eval()
with torch.no_grad():
    predicted_output = model(image_tensor)

predicted_segmentation = postprocess_output(predicted_output)

input_image_np = image_tensor.squeeze(0).squeeze(0).cpu().numpy()
analyzer.return_positions_experimental(input_image_np,predicted_segmentation)
# %%
import hyperspy.api as hs
import atomap.api as am
import numpy as np
from Analyzer import Analyzer
import cupy as cp
import pickle
import pandas as pd

with open('benchmark.pkl', 'rb') as f:
    data = pickle.load(f)
image_data = data["benchmark_sets"]
print(np.shape(image_data))
#%%
atomaps_positions_list = []
for i, group in enumerate(image_data):
    for j, image in enumerate(group):
            print(np.shape(image))
            image_data = image
            s = hs.signals.Signal2D(np.flipud(image_data))

            s_separation = am.get_feature_separation(s, separation_range=(4,7), threshold_rel=0.2)

            #s_separation.plot()

            atom_positions = am.get_atom_positions(s, 4, threshold_rel=0.3)

            sublattice = am.Sublattice(atom_position_list=atom_positions, image=s.data)
            sublattice.construct_zone_axes()
            sublattice.refine_atom_positions_using_center_of_mass(sublattice.image)  
            sublattice.refine_atom_positions_using_2d_gaussian(sublattice.image)  
            sublattice.plot()

            positions = sublattice.atom_positions
            atomaps_positions_list.append(positions)
#print(atomaps_positions_list)
#%%
from Analyzer import Analyzer
import cupy as cp
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet()
model = model.to(device)
loaded_data = torch.load("model_data_epoch_45.pth")

# Extract the epoch number, model state dictionary, and validation loss
#loaded_epoch = loaded_data['epoch']
loaded_model_state_dict = loaded_data['model_state_dict']
#loaded_validation_loss = loaded_data['validation_loss']
model.load_state_dict(loaded_model_state_dict)


outputs = []
unet_positions_list = []
for i, group in enumerate(image_data):
    for j, image in enumerate(group):
        image_cp = cp.array(image)
        normalized_image = cp.maximum((image_cp - image_cp.min()) / (image_cp.max() - image_cp.min()), 0)
        normalized_image_np = cp.asnumpy(normalized_image)  # Convert back to numpy for torch.tensor()
        normalized_image_np = normalized_image_np[np.newaxis, :, :]
        image_tensor = torch.tensor(normalized_image_np, dtype=torch.float32)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)
        model.eval()
        with torch.no_grad():
            predicted_output = model(image_tensor)
            predicted_segmentation = postprocess_output(predicted_output)
            analyzer = Analyzer()
            unet_positions = analyzer.return_positions_experimental(image,predicted_segmentation)
            outputs.append(predicted_segmentation)
            unet_positions_list.append(unet_positions)


point_sets = data["dataframes"]
positions = point_sets[0]
print(len(positions),len(unet_positions_list[30]))
analyzer = Analyzer()
unet_errors,atomaps_error = analyzer.compare_unet_atomaps(unet_positions_list[0],atomaps_positions_list, positions)

#
# print(len(data["dataframes"][0]["x","y"]),len(unet_positions_list[0]))
#analyzer = Analyzer()
#analyzer.set_image_list(data["benchmark_set"])
#analyzer.set_gt_list(data['exponential_ground_truths'])
#analyzer.set_pred_list(outputs)
#analyzer.set_point_sets(data["dataframes"])
#analyzer.set_SSIM_list(data["benchmark_ssim"])
#analyzer.set_atomaps_positions(atomaps_positions_list)
#comparison = analyzer.comparison_atomaps()

# %%
import statistics

unet_errors = comparison["unet_errors"]
atomaps_errors = comparison["atomaps_errors"]

mean_values_u = np.array([statistics.mean(np.array(sublist)[np.isfinite(sublist)]) for sublist in unet_errors])
standard_deviations_u = np.array([statistics.stdev(np.array(sublist)[np.isfinite(sublist)]) for sublist in unet_errors])
mean_values_a = np.array([statistics.mean(np.array(sublist)[np.isfinite(sublist)]) for sublist in atomaps_errors])
standard_deviations_a = np.array([statistics.stdev(np.array(sublist)[np.isfinite(sublist)]) for sublist in atomaps_errors])
ssim = np.array(comparison["SSIM"])
print(len(ssim),len(mean_values_u),len(standard_deviations_u),len(mean_values_a),len(standard_deviations_a))
print(ssim, mean_values_u, standard_deviations_u, mean_values_a, standard_deviations_a)
# Check the length of arrays
assert len(ssim) == len(mean_values_u) == len(standard_deviations_u), "Lengths of ssim, mean_values, and standard_deviations do not match"

# Sort the arrays according to the SSIM
def order_comparison(mean_values,standard_deviations,ssim):
    order = np.argsort(ssim)
    ordered_ssim = ssim[order]
    ordered_mean_values = mean_values[order]
    ordered_standard_deviations = standard_deviations[order]
    return ordered_ssim, ordered_mean_values, ordered_standard_deviations

a_ssim, a_means, a_stds = order_comparison(mean_values_a,standard_deviations_a,ssim)
assert len(ssim) == len(mean_values_u) == len(standard_deviations_u), "Lengths of ssim, mean_values, and standard_deviations do not match"
u_ssim, u_means, u_stds = order_comparison(mean_values_u,standard_deviations_u,ssim)
# %%
print(a_ssim, a_means, a_stds, u_ssim, u_means, u_stds)
fig, ax = plt.subplots()
ax.errorbar(u_ssim, u_means, yerr=u_stds, fmt='o', label='U-Net',c="c", capsize=3)

#
# ax.errorbar(a_ssim, a_means, yerr=a_stds, fmt='o', label='Gaussian fitting (atomaps)',c="orange", capsize=3)
#ax.plot(a_ssim[:-2], a_means[:-2], '-',c="orange")
ax.plot(u_ssim, u_means, '-',c="c")

# Dashed line for the last two segments
#ax.plot(a_ssim[-3:], a_means[-3:], '--',label='FNs & FPs' ,c="orange")

ax.legend()
ax.set_xlabel("SSIM")
ax.set_ylabel("Mean Error [pixels]")

# %%
