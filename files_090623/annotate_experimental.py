#%%
import matplotlib.pyplot as plt
import pickle
from PI_U_Net import UNet
from Analyzer import Analyzer
import os 
import glob
import cv2
import torch
import numpy as np

def preprocess_image(image_path):
    raw_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    normalized_image = np.maximum((raw_image - raw_image.min()) / (raw_image.max() - raw_image.min()), 0)
    normalized_image = normalized_image[np.newaxis, :, :]
    image_tensor = torch.tensor(normalized_image, dtype=torch.float32)
    return image_tensor.unsqueeze(0)

def postprocess_output(output_tensor):
    output_numpy = output_tensor.detach().cpu().numpy()
    output_image = np.squeeze(output_numpy)
    return output_image


#%%


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
analyzer = Analyzer()
unet = UNet()
unet = unet.to(device)
loaded_data = torch.load("best_model_data.pth")
loaded_model_state_dict = loaded_data['model_state_dict']
unet.load_state_dict(loaded_model_state_dict)

image_directory_path = "data/experimental_data/32bit/"
raw_images = []
point_sets = []
predicted_segmentations = []
for image_path in glob.glob(os.path.join(image_directory_path, "*.tif")):

    print(image_path)
    raw_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    raw_images.append(raw_image)
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    unet.eval()
    with torch.no_grad():
        predicted_output = unet(image_tensor)
        predicted_segmentation = postprocess_output(predicted_output)
        predicted_segmentations.append(predicted_segmentation)
    pred_positions = analyzer.return_positions_experimental(raw_image,predicted_segmentation)

    #plt.scatter(pred_positions[:, 1], pred_positions[:, 0])
    #plt.axis('equal')
    #plt.show()
    point_sets.append(pred_positions)


#%%

# initialize all labels as 0
labels = [np.zeros(len(point_set)) for point_set in point_sets]

for i, point_set in enumerate(point_sets):
    def on_click(event):
        # transform event coordinates to image coordinates
        ax = plt.gca()
        inv = ax.transData.inverted()
        event_in_image_coords = inv.transform((event.x, event.y))

        for j, point in enumerate(point_set):
            x, y = point
            if abs(x - event_in_image_coords[1]) < 1 and abs(y - event_in_image_coords[0]) < 1:  # adjust tolerance as needed
                # toggle label between 0 and 1
                labels[i][j] = 1 if labels[i][j] == 0 else 0

                # update the color of the points to visually distinguish them
                colors[j] = 'red' if labels[i][j] == 1 else 'blue'
                scatter.set_color(colors)

                plt.draw()  # redraw the plot

    fig, ax = plt.subplots()

    colors = ['blue' for _ in point_set]
    xs, ys = zip(*point_set)
    ax.imshow(raw_images[i], cmap='gray')
    scatter = ax.scatter(ys, xs, c=colors)

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.axis('equal')
    plt.show()
    plt.pause(0.1)  # Pause for user to close figure

# after labeling points, save points and labels to a dictionary
data = {'images':raw_images,'column_maps':predicted_segmentations,'points': point_sets, 'labels': labels}

# use pickle to save the dictionary to a file
with open('test_exp_data_2.pkl', 'wb') as f:
    pickle.dump(data, f)
# %%

# load the data
with open('test_exp_data_2.pkl', 'rb') as f:
    bata = pickle.load(f)

point_sets = bata['points']
images = bata['images']
labels = bata['labels']

# for each point set
for point_set, label, image in zip(point_sets, labels, images):
    # get points with label 1
    points_to_plot = point_set[label == 1]

    # plot points
    fig, ax = plt.subplots()
    xs, ys = zip(*points_to_plot)
    ax.imshow(image, cmap='gray')
    ax.scatter(ys, xs, c='c')

    plt.show()
    plt.pause(0.1)  # Pause for user to close figure
# %%
