#%%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import sys
import random
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import cv2
from scipy.ndimage import binary_closing
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence
import triangle
from scipy.interpolate import griddata
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.ndimage import median_filter
from scipy.spatial import Delaunay
from shapely.geometry import MultiPoint
from shapely.ops import unary_union
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import split

random.seed(1338)

class seg_UNet(nn.Module):
    def __init__(self):
        super(seg_UNet, self).__init__()

        def conv_block(in_channels, out_channels, dropout_rate=0.5):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
        )

        def upconv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels)
        )

        # Downsampling
        self.conv1 = conv_block(1, 128)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = conv_block(128, 256)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv3 = conv_block(256, 512)
        self.pool3 = nn.AvgPool2d(2, 2)
        self.conv4 = conv_block(512, 1024)
        self.pool4 = nn.AvgPool2d(2, 2)

        # Bridge
        self.conv5 = conv_block(1024, 2048)
        self.conv6 = conv_block(2048, 2048)

        # Upsampling
        self.upconv7 = upconv_block(2048, 1024)
        self.conv7 = conv_block(2048, 1024)
        self.dropout7 = nn.Dropout(0.3)
        self.upconv8 = upconv_block(1024, 512)
        self.conv8 = conv_block(1024, 512)
        self.dropout8 = nn.Dropout(0.3)
        self.upconv9 = upconv_block(512, 256)
        self.conv9 = conv_block(512, 256)
        self.dropout9 = nn.Dropout(0.3)
        self.upconv10 = upconv_block(256, 128)
        self.conv10 = conv_block(256, 128)

        self.output = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, img1):
        #x = torch.cat((img1, img2), dim=1)
        x = img1
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        x4 = self.pool2(x3)
        x5 = self.conv3(x4)
        x6 = self.pool3(x5)
        x7 = self.conv4(x6)
        x8 = self.pool4(x7)

        x9 = self.conv5(x8)
        x10 = self.conv6(x9)

        x11 = self.upconv7(x10)
        x12 = torch.cat([x11, x7], dim=1)
        x13 = self.conv7(x12)

        x14 = self.upconv8(x13)
        x15 = torch.cat([x14, x5], dim=1)
        x16 = self.conv8(x15)

        x17 = self.upconv9(x16)
        x18 = torch.cat([x17, x3], dim=1)
        x19 = self.conv9(x18)

        x20 = self.upconv10(x19)
        x21 = torch.cat([x20, x1], dim=1)
        x22 = self.conv10(x21)

        out = self.output(x22)

        return out
#%%
with open('large_dataset_with_predictions.pkl', 'rb') as f:
    data = pickle.load(f)

class CustomDataset(Dataset):
    def __init__(self,images,column_maps, ground_truths,positions,labels,density_maps,ldesc_maps):
        self.images = images
        self.ground_truths = ground_truths
        self.column_maps = column_maps
        self.positions = positions
        self.labels = labels
        self.density_maps = density_maps
        self.ldesc_maps = ldesc_maps

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        raw_image = self.images[idx]
        ground_truth = self.ground_truths[idx]
        #ground_truth = binary_closing(ground_truth, structure=np.ones((5,5))).astype(int)
        density_map = self.density_maps[idx]
        ldesc_map = self.ldesc_maps[idx]
        column_map = self.column_maps[idx]
        positions = self.positions[idx]
        labels = self.labels[idx]

        normalized_image = np.maximum((raw_image - raw_image.min()) / (raw_image.max() - raw_image.min()), 0)
        normalized_image = normalized_image[np.newaxis, :, :]
        ground_truth = ground_truth[np.newaxis, :, :]
        column_map = column_map[np.newaxis, :, :]
        density_map = density_map[np.newaxis, :, :]
        ldesc_map = ldesc_map[np.newaxis, :, :]
        image = torch.tensor(normalized_image, dtype=torch.float32)
        column_map = torch.tensor(column_map, dtype=torch.float32)
        ground_truth = torch.tensor(ground_truth, dtype=torch.float32)
        density_map = torch.tensor(density_map, dtype=torch.float32)
        ldesc_map = torch.tensor(ldesc_map, dtype=torch.float32)
        ground_truth = ground_truth * (column_map > 0.5)
        return image, ground_truth,column_map,positions,labels,density_map,ldesc_map

def plot_losses(train_losses, validation_losses):
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, validation_losses, label="Validation Loss")
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid()
    plt.show()

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

def compute_regions(output):
    mask = output > 0.5
    mask = mask.int()

    return mask

def label_points(positions, mask):

    mask = mask.squeeze()
    point_coordinates = positions
    pixel_coordinates = np.floor(point_coordinates).astype(int)
    
    point_labels = []
    for pixel_coord in pixel_coordinates:
        point_labels.append(mask[pixel_coord[0], pixel_coord[1]])
    point_labels = np.array(point_labels)
    return point_labels

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        num = 2. * (output * target).sum(dim=(2,3))
        den = output.sum(dim=(2,3)) + target.sum(dim=(2,3)) + self.eps

        return 1 - num / den

def iou_loss(pred, target):
    smooth = 1.  # Adds a smoothing factor to avoid division by zero

    # Flatten label and prediction tensors
    pred = pred.view(-1)
    target = target.view(-1)

    # Intersection is equivalent to True Positive count
    intersection = (pred * target).sum()

    # IoU formula
    total = (pred + target).sum()
    union = total - intersection 

    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU
positions = [df[['x','y']].to_numpy()*128 for df in data['dataframes']]
labels = [df['label'].to_numpy() for df in data['dataframes']]
#%%

def triangle_area(a, b, c):
    return 0.5 * abs(a[0]*b[1] + b[0]*c[1] + c[0]*a[1] - a[1]*b[0] - b[1]*c[0] - c[1]*a[0])

def calculate_density(tri, point, vertices):
    indices = np.where((tri.simplices == point).any(axis=1))[0]
    triangles = tri.simplices[indices]
    total_area = sum(triangle_area(vertices[triangle][0], vertices[triangle][1], vertices[triangle][2]) for triangle in triangles)
    neighbors = np.unique(triangles[triangles != point])
    density = total_area / (len(neighbors) + 1)
    return density

density_maps = []

for pos_index, pos in enumerate(positions):
    print(pos_index)
    tri = Delaunay(pos)
    densities = np.zeros(len(pos))
    for i in range(len(pos)):
        densities[i] = calculate_density(tri, i, pos)
    grid_x, grid_y = np.mgrid[0:128, 0:128]
    density_map = griddata(pos, densities, (grid_x, grid_y), method='cubic', fill_value=0)
    density_map = np.array(density_map.T)
    density_maps.append(density_map)
data['density_maps'] = density_maps 

#%%

def add_midpoints(pos, tri):
    #calculate the midpoint of all edges
    midpoints = []
    for i in range(len(tri.simplices)):
        #get the three points of the triangle
        p1 = pos[tri.simplices[i,0]]
        p2 = pos[tri.simplices[i,1]]
        p3 = pos[tri.simplices[i,2]]
        #calculate the midpoints of the edges
        m1 = (p1+p2)/2
        m2 = (p2+p3)/2
        m3 = (p3+p1)/2
        #add the midpoints to the list
        midpoints.append(m1)
        midpoints.append(m2)
        midpoints.append(m3)
    #convert the list to a numpy array
    midpoints = np.array(midpoints)
    #remove duplicate points
    midpoints = np.unique(midpoints, axis=0)
    #add the midpoints to the list of points
    pos = np.concatenate((pos,midpoints), axis=0)
    #calculate a new triangulation
    tri = Delaunay(pos)
    return pos, tri

from scipy.spatial import ConvexHull
from skimage.draw import polygon
from scipy.ndimage import median_filter
from collections import defaultdict

def voronoi_ridge_neighbors(vor):
    ridge_dict = defaultdict(list)
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        ridge_dict[tuple(sorted((p1, p2)))].extend([v1, v2])

    region_neighbors = defaultdict(set)
    for pr, vr in ridge_dict.items():
        r1, r2 = [vor.point_region[p] for p in pr]
        if all(v >= 0 for v in vr):  # ridge is finite
            region_neighbors[r1].add(r2)
            region_neighbors[r2].add(r1)
    
    return region_neighbors

def calc_lattice_descriptor_map(idx,positions):
    pos_original = positions[idx].copy()
    pos = positions[idx]
    tri = Delaunay(pos)
    pos, tri = add_midpoints(pos, tri)
    pos, tri = add_midpoints(pos, tri)
    vor = Voronoi(pos)

    neighbors = voronoi_ridge_neighbors(vor)
    sim_vals = []
    positions = []
    for region, neighbor_regions in neighbors.items():
        input_point_index = np.where(vor.point_region == region)[0]
        if len(input_point_index) == 0:
            continue
        input_point = vor.points[input_point_index[0]]
        positions.append(input_point)

        #get the points of the region
        points = vor.vertices[vor.regions[region]]
        if np.any(points > 128) or np.any(points < 0):
            #print('outside of image')
            sim_vals.append(0)
            continue
        #get the voronoi point of the region
        point_index = np.where(vor.point_region == region)[0]
        input_point = vor.points[point_index[0]]
        poly1 = Polygon(points)
        #get the neighboring points of the region
        all_ious = []  # Initialize all_ious here
        for nbor in neighbor_regions:
            #get the points of the neighboring region
            nbor_points = vor.vertices[vor.regions[nbor]]
            poly2 = Polygon(nbor_points)
            if np.any(nbor_points > 128) or np.any(nbor_points < 0) or not poly1.is_valid or not poly2.is_valid:
                #print('outside of image')
                continue
            #get the voronoi point of the neighboring region
            nbor_point_index = np.where(vor.point_region == nbor)[0]
            if len(nbor_point_index) == 0:
                continue
            nbor_input_point = vor.points[nbor_point_index[0]]
            #calculate the distance vector between the two voronoi points
            distance_vec = nbor_input_point - input_point
            #calculate the new points of the neighboring region
            nbor_points = nbor_points - distance_vec
            #calculate intersection over union
            poly2 = Polygon(nbor_points)
            intersection = poly1.intersection(poly2).area
            union = unary_union([poly1, poly2]).area
            iou = intersection / union if union else 0
            #print(iou)
            all_ious.append(iou)

        # Move the following lines out of the inner loop
        if len(all_ious) > 0:
            mean_IoU = np.mean(all_ious)
            #std_dev_IoU = np.std(all_ious)
            #cv_IoU = std_dev_IoU / mean_IoU
            sim_vals.append(mean_IoU)
        else:
            sim_vals.append(0)
    #print(len(sim_vals), len(positions))
    #make a grid interpolating between the positions and their corresponding similarity values
    grid_x, grid_y = np.mgrid[0:128, 0:128]
    lattice_descriptor_map = griddata(positions, sim_vals, (grid_x, grid_y), method='linear', fill_value=0)
    #calculate the convex hull of pos_sim, make a binary mask and multiply to grid_z
    hull = ConvexHull(pos_original)
    hull_points = pos_original[hull.vertices]
    x = hull_points[:, 1]
    y = hull_points[:, 0]
    mask = np.zeros_like(lattice_descriptor_map, dtype=bool)
    rr, cc = polygon(y, x)
    mask[rr, cc] = True
    lattice_descriptor_map = lattice_descriptor_map*mask
    # Define a 3x3 mean filter
    size =5
    # Apply mean filter to smooth the grid
    lattice_descriptor_map = median_filter(lattice_descriptor_map, size)
    dx, dy = np.gradient(lattice_descriptor_map)
    gradient_magnitude = np.sqrt(dx**2 + dy**2)
    
    return gradient_magnitude.T
#%%
plot_nr = 86
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(data['images'][plot_nr], cmap='gray') # add the origin parameter
plt.title('Image')
plt.axis('off')
lattice_descriptor_map = calc_lattice_descriptor_map(plot_nr, positions)
plt.subplot(1, 3, 3)
plt.imshow(lattice_descriptor_map, cmap='hot',vmin=0,vmax=0.3) # add the origin parameter
plt.scatter(positions[plot_nr][:, 0],positions[plot_nr][:, 1], s=3, c='g')
plt.axis('off')
plt.show()
#%%
lattice_descriptor_maps = []

for pos_index, pos in enumerate(positions):
    print(pos_index)
    lattice_descriptor_map = calc_lattice_descriptor_map(pos_index, positions)
    lattice_descriptor_maps.append(lattice_descriptor_map)
data['lattice_descriptor_maps'] = lattice_descriptor_maps 
#%%
#use pickle to save lattice_descriptor_maps
with open('lattice_descriptor_maps_grad.pkl', 'wb') as f:  # 'wb' stands for 'write bytes'
    pickle.dump(lattice_descriptor_maps, f)
#%%
#use pickle to load lattice_descriptor_maps
lattice_descriptor_maps = []
with open('lattice_descriptor_maps_grad.pkl', 'rb') as f:
    lattice_descriptor_maps = pickle.load(f)
data['lattice_descriptor_maps'] = lattice_descriptor_maps 
#%%
plot_nr =437
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(data['images'][plot_nr], cmap='gray') # add the origin parameter
plt.title('Image')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(density_maps[plot_nr], cmap='hot') # add the origin parameter
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(data['lattice_descriptor_maps'][plot_nr], cmap='hot') # add the origin parameter
plt.scatter(positions[plot_nr][:, 0],positions[plot_nr][:, 1], s=3, c='g')
plt.axis('off')
plt.show()

#%%

train_images, validation_images, train_ground_truths, validation_ground_truths, train_predictions, validation_predictions, train_positions, validation_positions, train_labels, validation_labels, train_dmaps,validation_dmaps, train_ldescriptor_maps, validation_ldescriptor_maps = train_test_split(
    data['images'], data['segmented_images'],data['predictions'], positions, labels,data['density_maps'],data['lattice_descriptor_maps'], test_size=0.25, random_state=1337)

train_dataset = CustomDataset(train_images,train_predictions, train_ground_truths,train_positions,train_labels,train_dmaps,train_ldescriptor_maps)
validation_dataset = CustomDataset(validation_images,validation_predictions, validation_ground_truths,validation_positions,validation_labels,validation_dmaps,validation_ldescriptor_maps)

def collate_fn(batch):
    images, ground_truths, column_maps, positions, labels, dmaps, ldesc_maps = zip(*batch)
    
    # Stack images, ground_truths, column_maps as usual
    images = torch.stack(images)
    ground_truths = torch.stack(ground_truths)
    column_maps = torch.stack(column_maps)
    dmaps = torch.stack(dmaps)
    ldesc_maps = torch.stack(ldesc_maps)

    # Padding variable-length sequences
    positions = pad_sequence([torch.tensor(pos) for pos in positions], batch_first=True, padding_value=129)
    labels = pad_sequence([torch.tensor(lab.astype(np.int64)) for lab in labels], batch_first=True, padding_value=129)
    
    return images, ground_truths, column_maps, positions, labels, dmaps, ldesc_maps

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate_fn)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn)

with open('test_exp_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

point_sets = test_data['points']
images = test_data['images']
preprocessed_images = []
exp_density_maps = []
exp_ldesc_maps = []
for idx, image in enumerate(images):
    normalized_image = np.maximum((image - image.min()) / (image.max() - image.min()), 0)
    normalized_image = normalized_image[np.newaxis,np.newaxis, :, :]
    image_tensor = torch.tensor(normalized_image, dtype=torch.float32)
    preprocessed_images.append(image_tensor)
    pos = point_sets[idx]
    tri = Delaunay(pos)
    densities = np.zeros(len(pos))
    for i in range(len(pos)):
        densities[i] = calculate_density(tri, i, pos)
    grid_x, grid_y = np.mgrid[0:128, 0:128]
    density_map = griddata(pos, densities, (grid_x, grid_y), method='cubic', fill_value=0)
    density_map = density_map.T
    density_map = density_map[np.newaxis,np.newaxis, :, :]
    density_map = torch.tensor(density_map, dtype=torch.float32)
    print(density_map.shape)
    exp_density_maps.append(density_map)
    ldesc_map = calc_lattice_descriptor_map(idx, point_sets)
    ldesc_map = ldesc_map[np.newaxis,np.newaxis, :, :]
    ldesc_map = torch.tensor(ldesc_map, dtype=torch.float32)
    exp_ldesc_maps.append(ldesc_map)
    

exp_images = preprocessed_images

column_maps = test_data['column_maps']
predictions = [torch.tensor(np.where(column_map >= 0.01, 1, 0)[np.newaxis,np.newaxis, :, :],dtype=torch.float32) for column_map in column_maps]
exp_column_maps = predictions
exp_labels = test_data['labels']
exp_point_sets=point_sets


image,column_map, ground_truth,positions,labels, density_map, ldesc_map = validation_dataset[0]
# %%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = seg_UNet().to(device)
#criterion= DiceLoss()
criterion = nn.BCELoss()
#criterion = iou_loss
optimizer = optim.Adam(model.parameters(), lr=0.000005)  # Starting learning rate is set to 0.1

# Define the scheduler

train_losses = []
train_accuracies = []
validation_losses = []
validation_accuracies= []
particle_accuracies = []
test_accuracies = []
tot_accuracies = []
num_epochs = 150
best_acc = 0.0
best_loss = np.inf
best_val_acc = 0.0
val_at_best_test = np.inf
checkp = 0

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.000001)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.00003)
for epoch in range(num_epochs):
    model.train()
    total_correct_train = 0
    total_predictions_train = 0
    total_correct_val = 0
    total_predictions_val = 0
    total_correct_1 = 0
    total_predictions_1 = 0
    running_train_accuracy = 0
    train_loss = 0.0

    for images, ground_truths, column_maps, positions, labels,density_maps,ldesc_maps in tqdm(train_dataloader):
        images = images.to(device)
        ground_truths = ground_truths.to(device)
        #print(column_maps)
        column_maps = column_maps.to(device)
        positions = positions.to(device)
        labels = labels.to(device)
        density_maps = density_maps.to(device)
        ldesc_maps = ldesc_maps.to(device)

        optimizer.zero_grad()
        #print(images.shape,column_maps.shape)
        #outputs = model(images, column_maps,density_maps,ldesc_maps)
        outputs = model(column_maps)
        predicted_regions = compute_regions(outputs)
        loss_bce = criterion(outputs, ground_truths)
        accuracy = 0
        tot_positions = 0
        for i in range(len(images)):
            positions_np = positions[i].cpu().numpy()
            positions_np = np.flip(positions_np,1)
            labels_np = labels[i].cpu().numpy()
            #print(len(positions_np),len(labels_np))
            # Create mask to ignore padded elements
            mask_pos = ~np.all(np.isclose(positions_np, 129.0), axis=-1)
            mask_lab = ~np.isclose(labels_np, 129.0)
            
            masked_positions = positions_np[mask_pos] - 0.5
            masked_labels = labels_np[mask_lab]
            #print(masked_positions.shape,masked_labels.shape)
            #if i==0 and epoch == 6:
            #    plt.imshow(predicted_regions[i].cpu().numpy().squeeze())
            #    plt.scatter(masked_positions[:,1],masked_positions[:,0])
            #    plt.show()
            pred_labels = label_points(masked_positions, predicted_regions[i].cpu().numpy())
            total_correct_train += accuracy_score(masked_labels, pred_labels, normalize=False)
            total_predictions_train += len(masked_positions)
            
        #print('batch accuracy',accuracy/tot_positions)
        #accuracy_loss = 1 - accuracy / len(images)
        loss = loss_bce
        loss.backward()
        optimizer.step()

        train_loss += loss_bce.item()
        running_train_accuracy += accuracy / len(images)

    train_loss /= len(train_dataloader)
    train_accuracy = running_train_accuracy / len(train_dataloader)
    train_acc = total_correct_train / total_predictions_train
    #print('train accuracy',train_acc)
    train_losses.append(train_loss)
    #print(train_accuracy)
    #train_accuracies.append(train_accuracy)

    model.eval()
    running_val_accuracy = 0
    validation_loss = 0.0
    image, ground_truth, column_map, output = None, None, None, None

    for i, (images, ground_truths, column_maps, positions, labels, density_maps, ldesc_maps) in enumerate(tqdm(validation_dataloader)):
        images = images.to(device)
        ground_truths = ground_truths.to(device)
        column_maps = column_maps.to(device)
        positions = positions.to(device)
        labels = labels.to(device)
        density_maps = density_maps.to(device)
        ldesc_maps = ldesc_maps.to(device)

        with torch.no_grad():
            #print(images.shape,column_maps.shape)
            #outputs = model(images, column_maps,density_maps,ldesc_maps)
            outputs = model(column_maps)
            #outputs = torch.where(outputs > 0.5, torch.tensor(1.0, device=outputs.device), torch.tensor(0.0, device=outputs.device))
            loss_bce = criterion(outputs, ground_truths)
            predicted_regions = compute_regions(outputs)
            accuracy = 0
            for j in range(len(images)):
                positions_np = positions[j].cpu().numpy()
                positions_np = np.flip(positions_np,1)
                labels_np = labels[j].cpu().numpy()
                
                # Create mask to ignore padded elements
                mask_pos = ~np.all(np.isclose(positions_np, 129.0), axis=-1)
                mask_lab = ~np.isclose(labels_np, 129.0)
                # Apply mask to positions, labels, and pixel_coordinates
                masked_positions = positions_np[mask_pos] - 0.5
                #print(masked_positions)
                #print(masked_positions.shape,positions[i].shape)
                masked_labels = labels_np[mask_lab]
                #total_predictions += len(masked_labels)
                total_predictions_1 += np.sum(masked_labels)

                pred_labels = label_points(masked_positions, predicted_regions[j].cpu().numpy())
                total_correct_val += accuracy_score(masked_labels, pred_labels, normalize=False)
                total_predictions_val += len(masked_positions)
                correct = (pred_labels == masked_labels)
                num_correct = np.sum(correct)
                #total_correct += num_correct

                correct_1 = (pred_labels == masked_labels) & (masked_labels == 1)
                num_correct_1 = np.sum(correct_1)
                total_correct_1 += num_correct_1

            accuracy_loss = 1 - accuracy / len(images)
            loss = loss_bce
            running_val_accuracy += accuracy / len(images)
            validation_loss += loss_bce.item()

            if i == 0:
                image = images[6].cpu().detach().numpy()[0]
                ground_truth = ground_truths[6].cpu().detach().numpy()[0]
                column_map = column_maps[6].cpu().detach().numpy()[0]
                ldesc_map = ldesc_maps[6].cpu().detach().numpy()[0]
                output = outputs[6].cpu().detach().numpy()[0]

    validation_loss /= len(validation_dataloader)
    val_acc = total_correct_val / total_predictions_val
    #for each image in test_data, compute only the total accuracy. test_data has 10 samples with image,column_map,point_set and corresponding label
    correct_class = 0
    total_points = 0
    for im,col_map,positions,labels,dens_map, ld_map in zip(exp_images, exp_column_maps, exp_point_sets, exp_labels, exp_density_maps, exp_ldesc_maps):
        with torch.no_grad():
            #print('hello')
            #print(image.shape,column_map.shape)
            im = im.to(device)
            col_map = col_map.to(device)
            dmap = dens_map.to(device)
            ld_map = ld_map.to(device)
            #out = model(im, col_map,dmap,ld_map)
            out = model(col_map)
            predicted_region = compute_regions(out)
            predicted_region = predicted_region.squeeze(0).squeeze(0).cpu().numpy()
            #print(positions.shape, predicted_region.shape)
            pixel_coordinates = np.floor(positions).astype(int)
            point_labels = []
            for pixel_coord in pixel_coordinates:
                point_labels.append(predicted_region[pixel_coord[0], pixel_coord[1]])
            pred_labels = np.array(point_labels)

            #print(pred_labels)
            correct = (pred_labels == labels)
            #print(correct)
            num_correct = np.sum(correct)
            #print(num_correct)
            correct_class += num_correct
            total_points += len(labels)
            #print(correct_class,total_points)
    test_accuracy = correct_class/total_points
    test_accuracies.append(test_accuracy)

    #overall_accuracy = total_correct / total_predictions
    overall_accuracy_1 = total_correct_1 / total_predictions_1
    
    #tot_accuracies.append(overall_accuracy)
    particle_accuracies.append(overall_accuracy_1)

    val_accuracy = running_val_accuracy / len(validation_dataloader)

    validation_losses.append(validation_loss)

    train_accuracies.append(train_acc)
    validation_accuracies.append(val_acc)

    if test_accuracy >= best_acc and validation_loss < val_at_best_test:
        best_acc = test_accuracy
        val_at_best_test = validation_loss
        checkp = epoch
        # Save the model state dictionary, best accuracy, and epoch number
        checkpoint = {
            'epoch': epoch,
            'best_accuracy': best_acc,
            'model_state_dict': model.state_dict()
        }
        torch.save(checkpoint, 'best_test_acc.pth')

    if validation_loss < best_loss:
        best_loss = validation_loss
        checkpoint = {
            'epoch': epoch,
            'best_loss': best_loss,
            'model_state_dict': model.state_dict()
        }
        torch.save(checkpoint, 'best_val_loss.pth')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        checkpoint = {
            'epoch': epoch,
            'best_val_acc': best_val_acc,
            'model_state_dict': model.state_dict()
        }
        torch.save(checkpoint, 'best_val_acc.pth')
        
    curr_lr = optimizer.param_groups[0]['lr']
    print('total_correct_train',total_correct_train,'total_predictions_train',total_predictions_train,'best_acc',best_acc, checkp)
    print(f'Epoch: {epoch + 1} \tLearning Rate: {curr_lr:.9f} \tTest accuracy: {test_accuracy:.4f} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {validation_loss:.4f} \tVal Accuracy: {val_acc:.4f} \tTrain Accuracy: {train_acc:.4f}')
    #scheduler.step()
    #plot only every 10 epoch
    if epoch % 10 == 0:
        plt.subplot(2,2,1)
        plt.imshow(image, cmap='gray')
        plt.title('Image')
        plt.axis('off')

        plt.subplot(2,2,2)
        plt.imshow(column_map, cmap='gray')
        plt.title('Column Map')
        plt.axis('off')

        plt.subplot(2,2,3)
        plt.imshow(ground_truth, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(2,2,4)
        plt.imshow(output, cmap='gray')
        plt.title('Output')
        plt.axis('off')

        plt.show() 
#save losses and accuracies
with open('train_losses.pkl', 'wb') as f:  # 'wb' stands for 'write bytes'
    pickle.dump(train_losses, f)
with open('train_accuracies.pkl', 'wb') as f:  # 'wb' stands for 'write bytes'
    pickle.dump(train_accuracies, f)
with open('validation_losses.pkl', 'wb') as f:  # 'wb' stands for 'write bytes'
    pickle.dump(validation_losses, f)
with open('validation_accuracies.pkl', 'wb') as f:  # 'wb' stands for 'write bytes'
    pickle.dump(validation_accuracies, f)
with open('test_accuracies.pkl', 'wb') as f:  # 'wb' stands for 'write bytes'
    pickle.dump(test_accuracies, f)


# Plotting the losses
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting the accuracies
plt.subplot(1, 3, 2)
plt.plot(range(1, num_epochs+1), validation_accuracies, label='Validation Accuracy')
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting the accuracies
plt.subplot(1, 3, 3)
plt.plot(range(1, num_epochs+1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
print("Best test accuracy: ", best_acc, "Best validation accuracy: ", best_val_acc, "Best loss: ", best_loss)

# %%
seg_data = torch.load("magiskt_bra.pth")

checkpoint = torch.load('best_val_acc.pth')

segmenter = seg_UNet()

from PI_U_Net import UNet
from Analyzer import Analyzer

localizer = UNet()
loc_data = torch.load("best_model_data.pth")
loaded_model_state_dict = loc_data['model_state_dict']
localizer.load_state_dict(loaded_model_state_dict)

#loaded_validation_loss = loaded_data['validation_loss']
segmenter.load_state_dict(seg_data)
#segmenter.load_state_dict(checkpoint['model_state_dict'])

image_path = "data/experimental_data/32bit/10.tif"
image_tensor = preprocess_image(image_path)
image_tensor = image_tensor.unsqueeze(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_tensor = image_tensor.to(device)
localizer = localizer.cuda()
# Pass the image tensor through the model
localizer.eval()
with torch.no_grad():
    predicted_output = localizer(image_tensor)

predicted_localization = postprocess_output(predicted_output)
prediction = torch.tensor(np.where(predicted_localization >= 0.01, 1, 0)[np.newaxis,np.newaxis, :, :],dtype=torch.float32)
predicted_localization_save = predicted_localization.copy()
analyzer = Analyzer()
pred_positions = analyzer.return_positions_experimental(image_tensor,predicted_localization)
#set all values larger than 0.1 in predicted_localization to 1
predicted_localization[predicted_localization > 0.02] = 1
input_image_np = image_tensor.squeeze(0).squeeze(0).cpu().numpy()

segmenter = segmenter.cuda()
segmenter.eval()

pos = pred_positions
tri = Delaunay(pos)
densities = np.zeros(len(pos))
for i in range(len(pos)):
    densities[i] = calculate_density(tri, i, pos)
grid_x, grid_y = np.mgrid[0:128, 0:128]
density_map = griddata(pos, densities, (grid_x, grid_y), method='cubic', fill_value=0)
density_map = density_map
density_map = density_map[np.newaxis,np.newaxis, :, :]
density_map = torch.tensor(density_map, dtype=torch.float32)
ld_map = calc_lattice_descriptor_map(0, [pos]).T
ld_map = ld_map[np.newaxis,np.newaxis, :, :]
ld_map = torch.tensor(ld_map, dtype=torch.float32)
#predicted_output[predicted_output > 0.01] = 1  
with torch.no_grad():
    print(image_tensor.shape,predicted_output.shape,density_map.shape)
    #predicted_output = segmenter(image_tensor, predicted_output,density_map.to(device),ld_map.to(device))
    predicted_output = segmenter(prediction.to(device))
predicted_segmentation = postprocess_output(predicted_output)

# Convert predicted segmentation to binary using threshold of 0.5
binary_segmentation = np.where(predicted_segmentation > 0.5, 1, 0)

labels = label_points(pred_positions, binary_segmentation)
plt.figure(figsize=(15, 5))
plt.imshow(input_image_np, cmap='gray')
plt.axis('off')
plt.show()
plt.figure(figsize=(15, 5))
plt.imshow(predicted_localization_save, cmap='gray')
plt.axis('off')
plt.show()
plt.figure(figsize=(15, 5))
plt.imshow(binary_segmentation, cmap='gray')
plt.axis('off')
plt.show()
plt.figure(figsize=(15, 5))
plt.imshow(input_image_np, cmap='gray')
for i in range(len(pos)):
    if labels[i] == 1:
        plt.scatter(pos[i][1],pos[i][0], s=40, c='springgreen')
    else:
        plt.scatter(pos[i][1],pos[i][0], s=40, c='darkorange')
plt.axis('off')
plt.show()
plt.figure(figsize=(5,5))
plt.scatter(pos[:,1],pos[:,0], s=40,c='k')
plt.xlim(0,128)
plt.ylim(0,128)
plt.gca().invert_yaxis()
plt.gca().set_xticks([])
plt.gca().set_yticks([])
#plt.axis('equal')
#plt.axis('off')
plt.show()
# Plot the input image and predicted segmentation
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(input_image_np, cmap='gray')
plt.axis('off')
plt.title('Input Image')

plt.subplot(1, 4, 2)
plt.imshow(predicted_localization, cmap='gray')
plt.axis('off')
plt.title('Predicted Localization')

plt.subplot(1, 4, 3)
plt.imshow(predicted_segmentation, cmap='gray')
plt.axis('off')
plt.title('Predicted Segmentation')

plt.subplot(1, 4, 4)
plt.imshow(binary_segmentation, cmap='gray')
plt.axis('off')
plt.title('Binary Segmentation')

plt.show()

#plot input image, predicted localization save, density map, binary segmentation, and Input image with pos on top colored by label
plt.figure(figsize=(15, 5))

plt.subplot(1, 6, 1)
plt.imshow(input_image_np, cmap='gray')
plt.axis('off')

plt.subplot(1, 6, 2)
plt.imshow(prediction.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray')
plt.axis('off')

plt.subplot(1, 6, 3)
plt.imshow(input_image_np, cmap='gray') # add the origin parameter
plt.triplot(pos[:,1], pos[:,0], tri.simplices)
plt.axis('off')

plt.subplot(1, 6, 4)
plt.imshow(ld_map.squeeze(0).squeeze(0).cpu().numpy(), cmap='hot', interpolation='nearest') # add the origin parameter
plt.axis('off')

plt.subplot(1, 6, 5)
plt.imshow(binary_segmentation, cmap='gray')
plt.axis('off')
#color by label, 1 is blue,0 is red


plt.subplot(1, 6, 6)
plt.imshow(input_image_np, cmap='gray')
for i in range(len(pos)):
    if labels[i] == 1:
        plt.scatter(pos[i][1],pos[i][0], s=10, c='c')
    else:
        plt.scatter(pos[i][1],pos[i][0], s=10, c='orange')
plt.axis('off')

plt.show()

#plot triangulation
#plt.figure(figsize=(10, 5))
#plt.imshow(input_image_np, cmap='gray') # add the origin parameter
#plt.triplot(pos[:,1], pos[:,0], tri.simplices)
#plt.axis('off')
#plt.show()

torch.cuda.empty_cache()

# %%
#Follow a single simulated example
from Data_Generator import Data_Generator
from PI_U_Net import UNet
from Analyzer import Analyzer
import pandas as pd
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN

seg_data = torch.load("magiskt_bra.pth")
checkpoint = torch.load('best_val_acc.pth')

segmenter = seg_UNet()
localizer = UNet()
loc_data = torch.load("best_model_data.pth")
loaded_model_state_dict = loc_data['model_state_dict']
localizer.load_state_dict(loaded_model_state_dict)
segmenter.load_state_dict(seg_data)

#number_of_images = 1
#dataset_name = "paper_img_xd"
#generator = Data_Generator()
#generator.generate_data(number_of_images,dataset_name)
with open('benchmark_100.pkl', 'rb') as f:
    data = pickle.load(f)
    
raw_image = data['images'][0]
normalized_image = np.maximum((raw_image - raw_image.min()) / (raw_image.max() - raw_image.min()), 0)
normalized_image = normalized_image[np.newaxis, :, :]
image_tensor = torch.tensor(normalized_image, dtype=torch.float32)
image_tensor = image_tensor.unsqueeze(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_tensor = image_tensor.to(device)
localizer = localizer.cuda()
# Pass the image tensor through the model
localizer.eval()
with torch.no_grad():
    predicted_output = localizer(image_tensor)
predicted_localization = postprocess_output(predicted_output)
predicted_localization_save = predicted_localization.copy()
prediction = torch.tensor(np.where(predicted_localization >= 0.01, 1, 0)[np.newaxis,np.newaxis, :, :],dtype=torch.float32)
analyzer = Analyzer()
pred_positions = analyzer.return_positions_experimental(image_tensor,predicted_localization)

analyzer.set_image_list(data["images"])
analyzer.set_gt_list(data['exponential_ground_truths'])
analyzer.set_pred_list([predicted_localization_save])
#print('boi',data["dataframes"])
analyzer.set_point_sets(data["dataframes"])
#mean, std = analyzer.calculate_average_error()

intersections,centers,radii,bbox_image,minr,minc,maxr,maxc,gt_pos=analyzer.plot_how_it_works(predicted_localization_save,30)
fig, ax = plt.subplots(figsize=(5,5))
ax.imshow(bbox_image, cmap='gray',origin='lower')

# Add circles
for center, radius in zip(centers, radii):
    circle = plt.Circle((center[1] - minc + 0.5, center[0] - minr + 0.5), radius, fill=False, edgecolor='darkorange', linewidth=4)
    ax.add_artist(circle)

if len(intersections) > 0:
    plt.scatter(intersections[:, 1]+0.5, intersections[:, 0]+0.5, marker='x',s=50, color='springgreen')
    plt.scatter(gt_pos[0]-minc+0.5, gt_pos[1]+0.5-minr, marker='o', s=60,color='red')
plt.xlim([0, maxc - minc])
plt.ylim([0, maxr - minr])
plt.axis('equal')
plt.axis('off')
plt.show()
gt = predicted_localization_save
fig, ax = plt.subplots()
ax.imshow(gt, cmap='gray',origin='lower')
# Create a Rectangle patch
print(maxc,minc,maxr,minr)
rect = patches.Rectangle((minc-2.5, minr-2.5), maxc - minc+4, maxr - minr+4, linewidth=4, edgecolor='darkorange', facecolor='none')
# Add the rectangle to the plot
ax.add_patch(rect)
ax.axis('off')
#ax.axis('equal')
# Show the figure
plt.show()

dbscan = DBSCAN(eps=0.1, min_samples=3)
dbscan.fit(intersections) 
labels = dbscan.labels_
clusters = [intersections[labels == i] for i in range(max(labels) + 1)]
clusters = sorted(clusters, key=len, reverse=True)

#get the brightest pixel in the bbox_image
flat_index = np.argmax(bbox_image)
row, col = np.unravel_index(flat_index, bbox_image.shape)

cluster_means = [np.mean(cluster, axis=0) for cluster in clusters]

def point_within_pixel(point, pixel):
    px_x, px_y = pixel
    pt_x, pt_y = point

    if pt_x >= px_x and pt_x <= px_x + 1 and pt_y >= px_y and pt_y <= px_y + 1:
        return True
    else:
        return False 
#------------------------------------------------------------------------------
# #implement in Analyzer + maybe look within 1 pixel euclidian distance instead of only in brightest pixel.    
# Find the index of the largest cluster whose mean lies within [row, col]
index = None
mean_value = None
print(cluster_means)
for i, mean in enumerate(cluster_means):
    print(mean,row,col)
    if point_within_pixel(mean+0.5, np.array([row, col])-0.5):
        print('yes')
        index = i
        mean_value = mean+0.5
        break
if mean_value is None:
    mean_value = np.array([row, col])
#------------------------------------------------------------------------------
print(mean_value)
predicted_column_position = mean_value
plt.figure()
plt.imshow(bbox_image,cmap='gray',origin='lower')
plt.scatter(intersections[:, 1]+0.5, intersections[:, 0]+0.5, marker='x',s=50, color='springgreen')
plt.scatter(np.array(cluster_means)[:, 1]+0.5, np.array(cluster_means)[:, 0]+0.5,s=50, marker='o', color='darkorange')
plt.scatter(predicted_column_position[1],predicted_column_position[0], s=60, c='k',marker='x')
plt.scatter(gt_pos[0]-minc+0.5, gt_pos[1]+0.5-minr, s=60, c='red',marker='o')
#plt.scatter(gt_pos[0]-minc+0.5, gt_pos[1]+0.5-minr, s=20, c='red')
plt.axis('off')
plt.show()

#set all values larger than 0.1 in predicted_localization to 1
predicted_localization[predicted_localization > 0.02] = 1
input_image_np = image_tensor.squeeze(0).squeeze(0).cpu().numpy()

segmenter = segmenter.cuda()
segmenter.eval()
with torch.no_grad():
    print(image_tensor.shape,predicted_output.shape,density_map.shape)
    #predicted_output = segmenter(image_tensor, predicted_output,density_map.to(device),ld_map.to(device))
    predicted_output = segmenter(prediction.to(device))

predicted_segmentation = postprocess_output(predicted_output)

# Convert predicted segmentation to binary using threshold of 0.5
binary_segmentation = np.where(predicted_segmentation > 0.5, 1, 0)

boi = binary_segmentation+predicted_localization*data["segmented_images"][0]
boi = np.where(boi ==2, 0,boi)

labels = label_points(pred_positions, binary_segmentation)
pos = pred_positions
plt.figure()
plt.imshow(input_image_np, cmap='gray', origin="lower")
plt.axis('off')
plt.show()
plt.figure()
plt.imshow(predicted_localization_save, cmap='gray', origin="lower")
plt.axis('off')
plt.show()
plt.figure()
plt.imshow(predicted_localization, cmap='gray', origin="lower")
plt.axis('off')
plt.show()
plt.figure()
plt.imshow(data["segmented_images"][0], cmap='gray', origin="lower")
plt.axis('off')
plt.show()
plt.figure()
plt.imshow(data["segmented_images"][0] * predicted_localization, cmap='gray', origin="lower")
plt.axis('off')
plt.show()
plt.figure()
plt.imshow(predicted_segmentation, cmap='gray',origin="lower")
plt.axis('off')
plt.show()
plt.figure()
plt.imshow(binary_segmentation, cmap='gray',origin="lower")
plt.axis('off')
plt.show()
plt.figure()
plt.imshow(input_image_np, cmap='gray',origin="lower")
for i in range(len(pos)):
    if labels[i] == 1:
        plt.scatter(pos[i][1],pos[i][0], s=20, c='springgreen')
    else:
        plt.scatter(pos[i][1],pos[i][0], s=20, c='darkorange')
plt.axis('off')
plt.show()
plt.figure()
plt.imshow(predicted_segmentation, cmap='gray',origin="lower")
for i in range(len(pos)):
    if labels[i] == 1:
        plt.scatter(pos[i][1]-0.5,pos[i][0]-0.5, s=20, c='springgreen')
    else:
        plt.scatter(pos[i][1]-0.5,pos[i][0]-0.5, s=20, c='darkorange')
plt.axis('off')
plt.show()
plt.figure()
plt.imshow(predicted_segmentation, cmap='gray',origin="lower")
for i in range(len(pos)):
    plt.scatter(pos[i][1]-0.5,pos[i][0]-0.5, s=20, c='darkorange')
plt.axis('off')
plt.show()
plt.figure()
plt.imshow(boi, cmap='gray',origin="lower")
plt.axis('off')
plt.show()
plt.figure()
plt.scatter(pos[:,1],pos[:,0], s=40,c='k')
plt.xlim(0,128)
plt.ylim(0,128)
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.gca().set_aspect('equal', adjustable='box')
#plt.axis('off')
plt.show()
#%%
import hyperspy.api as hs
import atomap.api as am
import statistics

image_data = data["benchmark_sets"]
print(image_data)

#extract x and y coordinates from dataframe
positions_gt = data["dataframes"][0][["x","y"]].to_numpy()*128
positions_gt = positions_gt[:,::-1]

unet_positions = []
unet_diff = []
atomaps_diff = []
tot = 0
import math
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def closest_pairs(set_a, set_b):
    set_a = [tuple(point) for point in set_a]
    set_b = [tuple(point) for point in set_b]
    
    closest_to_a = {}
    for point_a in set_a:
        min_distance = float('inf')
        closest_point = None
        for point_b in set_b:
            dist = euclidean_distance(point_a, point_b)
            if dist < min_distance and dist < 2:
                min_distance = dist
                closest_point = point_b
        if closest_point is not None:
            if closest_point not in closest_to_a or min_distance < euclidean_distance(closest_point, closest_to_a[closest_point]):
                closest_to_a[closest_point] = point_a

    # Points in set_a that don't have a close point in set_b
    unpaired_a = [point for point in set_a if point not in closest_to_a.values()]
    # Points in set_b that haven't been assigned
    unassigned_b = [point for point in set_b if point not in closest_to_a.keys()]
    #print(len(unpaired_a),len(unassigned_b))
    return unpaired_a + unassigned_b

for i, group in enumerate(image_data):
    if i == 0:
        for j, image in enumerate(group):
            raw_image = image
            normalized_image = np.maximum((raw_image - raw_image.min()) / (raw_image.max() - raw_image.min()), 0)
            normalized_image = normalized_image[np.newaxis, :, :]
            image_tensor = torch.tensor(normalized_image, dtype=torch.float32)
            image_tensor = image_tensor.unsqueeze(0)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            image_tensor = image_tensor.to(device)
            localizer = localizer.cuda()
            # Pass the image tensor through the model
            localizer.eval()
            with torch.no_grad():
                predicted_output = localizer(image_tensor)
            predicted_localization = postprocess_output(predicted_output)
            predicted_localization_save = predicted_localization.copy()
            prediction = torch.tensor(np.where(predicted_localization >= 0.01, 1, 0)[np.newaxis,np.newaxis, :, :],dtype=torch.float32)
            analyzer = Analyzer()
            pred_positions = analyzer.return_positions_experimental(image_tensor,predicted_localization)
            if j == 0:
                tot = len(pred_positions)
            unet_positions.append(pred_positions)
            #ignore points close to edge
            pred_positions = pred_positions[
                np.logical_and.reduce((
                    pred_positions[:,0] > 5, 
                    pred_positions[:,0] < 123, 
                    pred_positions[:,1] > 5, 
                    pred_positions[:,1] < 123
                ))
            ]
            positions_gt = positions_gt[
                np.logical_and.reduce((
                    positions_gt[:,0] > 5, 
                    positions_gt[:,0] < 123, 
                    positions_gt[:,1] > 5, 
                    positions_gt[:,1] < 123
                ))
            ]
            diff = closest_pairs(pred_positions,positions_gt)
            #print(len(diff))
            #plt.figure()
            #plot the tuples in diff as points
            #plt.scatter(positions_gt[:,0],positions_gt[:,1], s=40,c='k')
            #plt.scatter(np.array(diff)[:,0],np.array(diff)[:,1], s=40,c='r')
            #plt.show()
            unet_diff.append(len(diff))


atomaps_positions = []
for i, group in enumerate(image_data):
    if i == 0:
        for j, image in enumerate(group):
            image_data = image.T
            s = hs.signals.Signal2D(image_data)

            if j == 0:
                s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.3)
                atom_positions = am.get_atom_positions(s, 3, threshold_rel=0.3)
            if j == 1:
                s_separation = am.get_feature_separation(s, separation_range=(4,7), threshold_rel=0.2)
                atom_positions = am.get_atom_positions(s, 4, threshold_rel=0.2)
            if j == 2:
                s_separation = am.get_feature_separation(s, separation_range=(4,7), threshold_rel=0.2)
                atom_positions = am.get_atom_positions(s, 4, threshold_rel=0.2)
            if j == 3:
                s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.15)
                atom_positions = am.get_atom_positions(s, 4, threshold_rel=0.15)
            if j == 4:
                s_separation = am.get_feature_separation(s, separation_range=(4,7), threshold_rel=0.2)
                atom_positions = am.get_atom_positions(s, 4, threshold_rel=0.2)                
            if j == 5:
                s_separation = am.get_feature_separation(s, separation_range=(4,7), threshold_rel=0.2)
                atom_positions = am.get_atom_positions(s, 5, threshold_rel=0.2)
            if j == 6:
                s_separation = am.get_feature_separation(s, separation_range=(4,7), threshold_rel=0.2)
                atom_positions = am.get_atom_positions(s, 4, threshold_rel=0.2)
            if j == 7:
                s_separation = am.get_feature_separation(s, separation_range=(4,7), threshold_rel=0.2)
                atom_positions = am.get_atom_positions(s, 4, threshold_rel=0.2)
            if j == 8:
                s_separation = am.get_feature_separation(s, separation_range=(4,7), threshold_rel=0.2)
                atom_positions = am.get_atom_positions(s, 5, threshold_rel=0.2)
            if j == 9:
                s_separation = am.get_feature_separation(s, separation_range=(4,7), threshold_rel=0.2)
                atom_positions = am.get_atom_positions(s, 6, threshold_rel=0.2)
            if j == 10:
                s_separation = am.get_feature_separation(s, separation_range=(4,7), threshold_rel=0.2)
                atom_positions = am.get_atom_positions(s, 6, threshold_rel=0.2)
            if j == 11:
                s_separation = am.get_feature_separation(s, separation_range=(4,7), threshold_rel=0.15)
                atom_positions = am.get_atom_positions(s, 6, threshold_rel=0.15)

            sublattice = am.Sublattice(atom_position_list=atom_positions, image=s.data)
            sublattice.construct_zone_axes()
            sublattice.refine_atom_positions_using_center_of_mass(sublattice.image)  
            sublattice.refine_atom_positions_using_2d_gaussian(sublattice.image)  
            #sublattice.plot()
            positions_atomaps = sublattice.atom_positions
            if j == 0:
                tot = len(positions_atomaps)
            atomaps_positions.append(positions_atomaps)
            #ignore points close to edge
            positions_atomaps = positions_atomaps[
                np.logical_and.reduce((
                    positions_atomaps[:,0] > 5, 
                    positions_atomaps[:,0] < 123, 
                    positions_atomaps[:,1] > 5, 
                    positions_atomaps[:,1] < 123
                ))
            ]
            positions_gt = positions_gt[
                np.logical_and.reduce((
                    positions_gt[:,0] > 5, 
                    positions_gt[:,0] < 123, 
                    positions_gt[:,1] > 5, 
                    positions_gt[:,1] < 123
                ))
            ]
            diff = closest_pairs(positions_atomaps,positions_gt)            
            atomaps_diff.append(len(diff))

unet_errors,atomaps_errors = analyzer.compare_unet_atomaps(unet_positions,atomaps_positions, positions_gt)
mean_values_u = np.array([statistics.mean(np.array(sublist)[np.isfinite(sublist)]) for sublist in unet_errors])
standard_deviations_u = np.array([statistics.stdev(np.array(sublist)[np.isfinite(sublist)]) for sublist in unet_errors])
mean_values_a = np.array([statistics.mean(np.array(sublist)[np.isfinite(sublist)]) for sublist in atomaps_errors])
standard_deviations_a = np.array([statistics.stdev(np.array(sublist)[np.isfinite(sublist)]) for sublist in atomaps_errors])
ssims = data["benchmark_ssims"][0]
#&&
plt.figure()
plt.scatter(positions_gt[:,0],positions_gt[:,1], s=40,c='k')
plt.scatter(atomaps_positions[11][:,0],atomaps_positions[11][:,1], s=40,c='r')
plt.scatter(unet_positions[11][:,0],unet_positions[11][:,1], s=40,c='b')
plt.xlim(0,128)
plt.ylim(0,128)
#%%
unet_errors,atomaps_errors = analyzer.compare_unet_atomaps(unet_positions,atomaps_positions, positions_gt)
print(atomaps_errors[0])
#%%
import statistics
mean_values_u = np.array([statistics.mean(np.array(sublist)[np.isfinite(sublist)]) for sublist in unet_errors])
standard_deviations_u = np.array([statistics.stdev(np.array(sublist)[np.isfinite(sublist)]) for sublist in unet_errors])
mean_values_a = np.array([statistics.mean(np.array(sublist)[np.isfinite(sublist)]) for sublist in atomaps_errors])
standard_deviations_a = np.array([statistics.stdev(np.array(sublist)[np.isfinite(sublist)]) for sublist in atomaps_errors])
ssims = data["benchmark_ssims"][0]

#use plt errorbar to plot mean_values with standard deviations as errorbars, ssim is shared x-axis
plt.figure()
plt.errorbar(ssims, mean_values_u, standard_deviations_u, linestyle='None', marker='s')
plt.errorbar(ssims, mean_values_a, standard_deviations_a, linestyle='None', marker='^')
#plt.errorbar(ssims, unet_diff, linestyle='None', marker='o')
#plt.errorbar(ssims, atomaps_diff, linestyle='None', marker='o')
plt.xlabel('SSIM')
plt.ylabel('Mean error')
plt.legend(['UNET', 'Atomap'])
plt.show()

fig, ax1 = plt.subplots()

# Plotting data on primary y-axis
ax1.errorbar(ssims, mean_values_a, standard_deviations_a, linestyle='-', marker='s', color='darkorange',capsize=2,capthick=2, markerfacecolor='darkorange', markeredgecolor='black')
ax1.errorbar(ssims, mean_values_u, standard_deviations_u, linestyle='-', marker='o', color='darkgreen',capsize=2,capthick=2, markerfacecolor='darkgreen', markeredgecolor='black')
ax1.set_ylabel('Mean error [pixels]', color='k')
ax1.tick_params(axis='y', labelcolor='k')
ax1.set_xlabel('SSIM')
ax1.legend(['Atomaps','U-Net'], loc='lower left')
# Create a twin y-axis
ax2 = ax1.twinx()

# Plotting data on secondary y-axis
ax2.plot(ssims, atomaps_diff, linestyle=':', color='gray')
ax2.plot(ssims, unet_diff, linestyle='--', color='gray')

# Set secondary y-axis color
ax2.set_ylabel('# False predictions', color='gray')
ax2.tick_params(axis='y', labelcolor='gray')
ax2.set_yticks(np.arange(-7, 8, step=2))
ax2.legend(['Atomaps','U-Net'], loc='upper right')

#plot all images in data["benchmark_sets"][0] in a 3x4 grid with SSIM (two value digits) in the lower left of the image, i dont want any white space between images
plt.figure(figsize=(11,14.75))
for i, image in enumerate(data["benchmark_sets"][0]):
    ax = plt.subplot(4, 3, i+1)
    ax.imshow(image, cmap='gray', origin="lower")
    ax.axis('off')
    ax.set_aspect('equal')  # Ensure the aspect ratio is equal
    # Use plt.text to place the SSIM on the image
    plt.text(5, 10, "SSIM: {:.2f}".format(data["benchmark_ssims"][0][i]), 
             color='white', backgroundcolor='black', fontsize=15)

plt.subplots_adjust(wspace=0, hspace=0)  # Set spacing to zero
plt.show()

plt.subplot(1,3,1)
plt.imshow(data["benchmark_sets"][0][0], cmap='gray',origin="lower")
plt.scatter(unet_positions[0][:,1],unet_positions[0][:,0], s=10,c='r')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(data["benchmark_sets"][0][1], cmap='gray',origin="lower")
plt.scatter(unet_positions[0][:,1],unet_positions[0][:,0], s=10,c='r')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(data["benchmark_sets"][0][3], cmap='gray',origin="lower")
plt.scatter(atomaps_positions[3][:,1],atomaps_positions[3][:,0], s=10,c='r')
plt.axis('off')


# %%
import hyperspy.api as hs
import atomap.api as am
import statistics
means_u = []
stds_u = []
mean_a = []
stds_a = []
ssims = []
tot_columns = []
tot_found_columns = []
differences = []

with open('benchmark_100.pkl', 'rb') as f:
    data = pickle.load(f)
print("i")
image_data = data["benchmark_sets"]

for i, group in enumerate(image_data):
    print(i)
    unet_positions = []
    atomaps_positions = []

    positions_gt = data["dataframes"][i][["x","y"]].to_numpy()*128
    positions_gt = positions_gt[:,::-1]
    tot_columns = 0
    diff = []
    for j, image in enumerate(group):
        raw_image = image
        normalized_image = np.maximum((raw_image - raw_image.min()) / (raw_image.max() - raw_image.min()), 0)
        normalized_image = normalized_image[np.newaxis, :, :]
        image_tensor = torch.tensor(normalized_image, dtype=torch.float32)
        image_tensor = image_tensor.unsqueeze(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.to(device)
        localizer = localizer.cuda()
        # Pass the image tensor through the model
        localizer.eval()
        with torch.no_grad():
            predicted_output = localizer(image_tensor)
        predicted_localization = postprocess_output(predicted_output)
        predicted_localization_save = predicted_localization.copy()
        prediction = torch.tensor(np.where(predicted_localization >= 0.01, 1, 0)[np.newaxis,np.newaxis, :, :],dtype=torch.float32)
        analyzer = Analyzer()
        pred_positions = analyzer.return_positions_experimental(image_tensor,predicted_localization)
        if j == 0:
            tot_columns = len(pred_positions)
        #ignore points close to edge
        pred_positions = pred_positions[
            np.logical_and.reduce((
                pred_positions[:,0] > 5, 
                pred_positions[:,0] < 123, 
                pred_positions[:,1] > 5, 
                pred_positions[:,1] < 123
            ))
        ]
        positions_gt = positions_gt[
            np.logical_and.reduce((
                positions_gt[:,0] > 5, 
                positions_gt[:,0] < 123, 
                positions_gt[:,1] > 5, 
                positions_gt[:,1] < 123
            ))
        ]
        diffs = closest_pairs(pred_positions,positions_gt)
        diff.append(len(diffs))
        unet_positions.append(pred_positions)
        positions_atomaps = pred_positions.copy()
        atomaps_positions.append(positions_atomaps)
    differences.append(diff)
    
    unet_errors = []
    atomaps_errors = []
    for positions_unet,positions_atomaps in zip(unet_positions,atomaps_positions):
        nn_unet = analyzer.assign_nearest(positions_unet,positions_gt,3)
        nn_atomaps = analyzer.assign_nearest(positions_atomaps,positions_gt,3)
        unet_error = analyzer.calculate_error(positions_unet,nn_unet)
        atomaps_error = analyzer.calculate_error(positions_atomaps,nn_atomaps)
        unet_errors.append(unet_error)
        atomaps_errors.append(atomaps_error)

    #print(unet_errors)
    mean_values_u = np.array([statistics.mean(np.array(sublist)[np.isfinite(sublist)]) for sublist in unet_errors])
    #print(mean_values_u)
    standard_deviations_u = np.array([statistics.stdev(np.array(sublist)[np.isfinite(sublist)]) for sublist in unet_errors])
    print(mean_values_u,standard_deviations_u)
    #mean_values_a = np.array([statistics.mean(np.array(sublist)[np.isfinite(sublist)]) for sublist in atomaps_errors])
    #standard_deviations_a = np.array([statistics.stdev(np.array(sublist)[np.isfinite(sublist)]) for sublist in atomaps_errors])
    ssim = data["benchmark_ssims"][i]
    #if i == 9:
        #remove the last two elements of ssim (bug)
    #    ssim = ssim[:-2]
    print(ssim)
    means_u.append(mean_values_u)  
    stds_u.append(standard_deviations_u)
    ssims.append(ssim)
print(len(means_u))
#mean_a.append(mean_values_a)
#stds_a.append(standard_deviations_a)
#print(means_u)
# %%
#average the means and stds in means_u and stds_u
mean_values_u = np.sum(means_u,axis=0)/len(means_u)
standard_deviations_u = np.sum(stds_u,axis=0)/len(stds_u)
ssim_values = np.sum(ssims,axis=0)/len(ssims)
from scipy.stats import mode
mean_diff = np.sum(differences,axis=0)/len(differences)
median_diff = np.median(differences,axis=0)
mode_diff, counts = mode(differences, axis=0) 
print(mode_diff, counts)

fig, ax1 = plt.subplots()

# Plotting data on primary y-axis
ax1.errorbar(ssim_values, mean_values_u, standard_deviations_u, linestyle='-', marker='o', color='darkgreen',capsize=2,capthick=2, markerfacecolor='darkgreen', markeredgecolor='black')
ax1.set_ylabel('Mean error [pixels]', color='darkgreen', weight='bold')
ax1.tick_params(axis='y', labelcolor='darkgreen')
ax1.tick_params(axis='both', labelsize=11)
ax1.set_xlabel('SSIM',weight='bold')
ax1.legend(['U-Net'], facecolor='white', edgecolor='black', loc=(0.57,0.9125),prop={'weight':'bold'})
# Create a twin y-axis
ax2 = ax1.twinx()

# Plotting data on secondary y-axis
ax2.plot(ssim_values, mean_diff, linestyle='-', color='gray', markerfacecolor='gray')
long_dotted_line = (0, (1, 7))  # 1 unit dot, 5 units space

#ax2.axhline(y=1, color='k', linestyle="-",label="_nolegend_")
ax2.plot(ssim_values, median_diff, linestyle='--', color='gray')
ax2.plot(ssim_values, mode_diff[0], linestyle=':', color='gray')

# Set secondary y-axis color
ax2.set_ylabel('# False predictions', color='gray',weight='bold')
ax2.tick_params(axis='y', labelcolor='gray')
ax2.tick_params(axis='y', labelsize=11)
ax2.set_yticks(np.arange(0, 4, step=1))
ax2.legend(['Mean','Median', 'Mode'],facecolor='white', edgecolor='black', loc='upper right',prop={'weight':'bold'})

plt.show()
#%%
for i in range(len(means_u)):
    plt.figure()
    #if i == 9:
    #    for idx, item in enumerate(ssims[i]):
    #        print(idx, item)
    print(i,len(ssims[i]),len(means_u[i]),len(stds_u[i]))
    plt.errorbar(ssims[i], means_u[i], stds_u[i], linestyle='None', marker='^')
    #plt.errorbar(ssims[i], mean_a[i], stds_a[i], linestyle='None', marker='^')
    plt.xlabel('SSIM')
    plt.ylabel('Mean error')
    plt.legend(['UNET', 'Atomap'])
    plt.show()
# %%
