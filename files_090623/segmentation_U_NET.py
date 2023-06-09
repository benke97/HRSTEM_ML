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


random.seed(1337)

class seg_UNet(nn.Module):
    def __init__(self):
        super(seg_UNet, self).__init__()

        def conv_block(in_channels, out_channels, dropout_rate=0.5):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            )

        def upconv_block(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # Downsampling
        self.conv1 = conv_block(3, 128)
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

    def forward(self, img1, img2, img3):
        x = torch.cat((img1, img2, img3), dim=1)
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

with open('dataset_workstation_3_with_predictions.pkl', 'rb') as f:
    data = pickle.load(f)

class CustomDataset(Dataset):
    def __init__(self,images,column_maps, ground_truths,positions,labels,density_maps):
        self.images = images
        self.ground_truths = ground_truths
        self.column_maps = column_maps
        self.positions = positions
        self.labels = labels
        self.density_maps = density_maps

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        raw_image = self.images[idx]
        ground_truth = self.ground_truths[idx]
        #ground_truth = binary_closing(ground_truth, structure=np.ones((5,5))).astype(int)
        density_map = self.density_maps[idx]
        column_map = self.column_maps[idx]
        positions = self.positions[idx]
        labels = self.labels[idx]

        normalized_image = np.maximum((raw_image - raw_image.min()) / (raw_image.max() - raw_image.min()), 0)
        normalized_image = normalized_image[np.newaxis, :, :]
        ground_truth = ground_truth[np.newaxis, :, :]
        column_map = column_map[np.newaxis, :, :]
        density_map = density_map[np.newaxis, :, :]
        image = torch.tensor(normalized_image, dtype=torch.float32)
        column_map = torch.tensor(column_map, dtype=torch.float32)
        ground_truth = torch.tensor(ground_truth, dtype=torch.float32)
        density_map = torch.tensor(density_map, dtype=torch.float32)
        #ground_truth = ground_truth * (column_map > 0.5)
        return image, ground_truth,column_map,positions,labels,density_map

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
#%%
positions = [df[['x','y']].to_numpy()*128 for df in data['dataframes']]
labels = [df['label'].to_numpy() for df in data['dataframes']]
#%%
from scipy.spatial import Delaunay
from scipy.interpolate import griddata

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

# Plot the filtered triangulation
#make this a 1,2 subplot with image and density map
#make this a 1,2 subplot with image and density map
#plot image with positions on top

plt.figure(figsize=(10, 5))
plt.imshow(data['images'][23], cmap='gray') # add the origin parameter
plt.scatter(positions[23][:, 0],positions[23][:, 1], s=1, c='r')
plt.show()


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(data['images'][34], cmap='gray') # add the origin parameter
plt.title('Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(density_maps[34], cmap='hot', interpolation='nearest') # add the origin parameter
plt.axis('off')
plt.show()
#%%

train_images, validation_images, train_ground_truths, validation_ground_truths, train_predictions, validation_predictions, train_positions, validation_positions, train_labels, validation_labels, train_dmaps,validation_dmaps = train_test_split(
    data['images'], data['segmented_images'],data['predictions'], positions, labels,data['density_maps'], test_size=0.25, random_state=1337)

train_dataset = CustomDataset(train_images,train_predictions, train_ground_truths,train_positions,train_labels,train_dmaps)
validation_dataset = CustomDataset(validation_images,validation_predictions, validation_ground_truths,validation_positions,validation_labels,validation_dmaps)

def collate_fn(batch):
    images, ground_truths, column_maps, positions, labels, dmaps = zip(*batch)
    
    # Stack images, ground_truths, column_maps as usual
    images = torch.stack(images)
    ground_truths = torch.stack(ground_truths)
    column_maps = torch.stack(column_maps)
    dmaps = torch.stack(dmaps)
    
    # Padding variable-length sequences
    positions = pad_sequence([torch.tensor(pos) for pos in positions], batch_first=True, padding_value=129)
    labels = pad_sequence([torch.tensor(lab.astype(np.int64)) for lab in labels], batch_first=True, padding_value=129)
    
    return images, ground_truths, column_maps, positions, labels,dmaps

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate_fn)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn)

with open('test_exp_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

point_sets = test_data['points']
images = test_data['images']
preprocessed_images = []
exp_density_maps = []
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

exp_images = preprocessed_images

column_maps = test_data['column_maps']
predictions = [torch.tensor(np.where(column_map >= 0.01, 1, 0)[np.newaxis,np.newaxis, :, :],dtype=torch.float32) for column_map in column_maps]
exp_column_maps = predictions
exp_labels = test_data['labels']
exp_point_sets=point_sets


image,column_map, ground_truth,positions,labels, density_map = validation_dataset[0]
# %%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = seg_UNet().to(device)
#criterion= DiceLoss()
criterion = nn.BCELoss()
#criterion = iou_loss
optimizer = optim.Adam(model.parameters(), lr=0.000075)  # Starting learning rate is set to 0.1

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

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00001)

for epoch in range(num_epochs):
    model.train()
    total_correct = 0
    total_predictions = 0
    total_correct_1 = 0
    total_predictions_1 = 0
    running_train_accuracy = 0
    train_loss = 0.0

    for images, ground_truths, column_maps, positions, labels,density_maps in tqdm(train_dataloader):
        images = images.to(device)
        ground_truths = ground_truths.to(device)
        #print(column_maps)
        column_maps = column_maps.to(device)
        positions = positions.to(device)
        labels = labels.to(device)
        density_maps = density_maps.to(device)

        optimizer.zero_grad()
        #print(images.shape,column_maps.shape)
        outputs = model(images, column_maps,density_maps)
        predicted_regions = compute_regions(outputs)
        loss_bce = criterion(outputs, ground_truths)
        accuracy = 0

        for i in range(len(images)):
            positions_np = positions[i].cpu().numpy()
            labels_np = labels[i].cpu().numpy()
            
            # Create mask to ignore padded elements
            mask_pos = ~np.all(np.isclose(positions_np, 129.0), axis=-1)
            mask_lab = ~np.isclose(labels_np, 129.0)
            
            masked_positions = positions_np[mask_pos]
            masked_labels = labels_np[mask_lab]

            pred_labels = label_points(masked_positions, predicted_regions[i].cpu().numpy())
            accuracy += accuracy_score(masked_labels, pred_labels)

        accuracy_loss = 1 - accuracy / len(images)
        loss = loss_bce
        loss.backward()
        optimizer.step()

        train_loss += loss_bce.item()
        running_train_accuracy += accuracy / len(images)

    train_loss /= len(train_dataloader)
    train_accuracy = running_train_accuracy / len(train_dataloader)
    train_losses.append(train_loss)
    #print(train_accuracy)
    train_accuracies.append(train_accuracy)

    model.eval()
    running_val_accuracy = 0
    validation_loss = 0.0
    image, ground_truth, column_map, output = None, None, None, None

    for i, (images, ground_truths, column_maps, positions, labels, density_maps) in enumerate(tqdm(validation_dataloader)):
        images = images.to(device)
        ground_truths = ground_truths.to(device)
        column_maps = column_maps.to(device)
        positions = positions.to(device)
        labels = labels.to(device)
        density_maps = density_maps.to(device)

        with torch.no_grad():
            #print(images.shape,column_maps.shape)
            outputs = model(images, column_maps,density_maps)
            #outputs = torch.where(outputs > 0.5, torch.tensor(1.0, device=outputs.device), torch.tensor(0.0, device=outputs.device))
            loss_bce = criterion(outputs, ground_truths)
            predicted_regions = compute_regions(outputs)
            accuracy = 0
            for j in range(len(images)):
                positions_np = positions[j].cpu().numpy()
                labels_np = labels[j].cpu().numpy()
                
                # Create mask to ignore padded elements
                mask_pos = ~np.all(np.isclose(positions_np, 129.0), axis=-1)
                mask_lab = ~np.isclose(labels_np, 129.0)
                
                # Apply mask to positions, labels, and pixel_coordinates
                masked_positions = positions_np[mask_pos]
                #print(masked_positions)
                #print(masked_positions.shape,positions[i].shape)
                masked_labels = labels_np[mask_lab]
                total_predictions += len(masked_labels)
                total_predictions_1 += np.sum(masked_labels)

                pred_labels = label_points(masked_positions, predicted_regions[j].cpu().numpy())
                
                correct = (pred_labels == masked_labels)
                num_correct = np.sum(correct)
                total_correct += num_correct

                correct_1 = (pred_labels == masked_labels) & (masked_labels == 1)
                num_correct_1 = np.sum(correct_1)
                total_correct_1 += num_correct_1

            accuracy_loss = 1 - accuracy / len(images)
            loss = loss_bce
            running_val_accuracy += accuracy / len(images)
            validation_loss += loss_bce.item()

            if i == 0:
                image = images[5].cpu().detach().numpy()[0]
                ground_truth = ground_truths[5].cpu().detach().numpy()[0]
                column_map = column_maps[5].cpu().detach().numpy()[0]
                output = outputs[5].cpu().detach().numpy()[0]

    validation_loss /= len(validation_dataloader)

    #for each image in test_data, compute only the total accuracy. test_data has 10 samples with image,column_map,point_set and corresponding label
    correct_class = 0
    total_points = 0
    for im,col_map,positions,labels,dens_map in zip(exp_images, exp_column_maps, exp_point_sets, exp_labels, exp_density_maps):
        with torch.no_grad():
            #print('hello')
            #print(image.shape,column_map.shape)
            im = im.to(device)
            col_map = col_map.to(device)
            dmap = dens_map.to(device)
            out = model(im, col_map,dmap)
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

    overall_accuracy = total_correct / total_predictions
    overall_accuracy_1 = total_correct_1 / total_predictions_1
    
    tot_accuracies.append(overall_accuracy)
    particle_accuracies.append(overall_accuracy_1)

    val_accuracy = running_val_accuracy / len(validation_dataloader)

    validation_losses.append(validation_loss)
    validation_accuracies.append(val_accuracy)

    if test_accuracy > best_acc:
        best_acc = test_accuracy
        torch.save(model.state_dict(), 'best_model.pth')
    #scheduler.step()
    curr_lr = optimizer.param_groups[0]['lr']

    print(f'Epoch: {epoch + 1} \tLearning Rate: {curr_lr:.9f} \tTest accuracy: {test_accuracy:.4f} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {validation_loss:.4f} \tVal Accuracy: {overall_accuracy:.4f} \tVal Particle Accuracy: {overall_accuracy_1:.4f}')
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
plt.plot(range(1, num_epochs+1), tot_accuracies, label='Total Accuracy')
plt.plot(range(1, num_epochs+1), particle_accuracies, label='Particle Accuracy')
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
print(best_acc)

# %%
seg_data = torch.load("98percent_acc.pth")
segmenter = seg_UNet()

from PI_U_Net import UNet
from Analyzer import Analyzer

localizer = UNet()
loc_data = torch.load("best_model_data.pth")
loaded_model_state_dict = loc_data['model_state_dict']
localizer.load_state_dict(loaded_model_state_dict)

#loaded_validation_loss = loaded_data['validation_loss']
segmenter.load_state_dict(seg_data)

image_path = "data/experimental_data/32bit/01.tif"
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
#predicted_output[predicted_output > 0.01] = 1  
with torch.no_grad():
    print(image_tensor.shape,predicted_output.shape,density_map.shape)
    predicted_output = segmenter(image_tensor, predicted_output,density_map.to(device))

predicted_segmentation = postprocess_output(predicted_output)

# Convert predicted segmentation to binary using threshold of 0.5
binary_segmentation = np.where(predicted_segmentation > 0.5, 1, 0)

labels = label_points(pred_positions, binary_segmentation)
print
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
plt.imshow(predicted_localization_save, cmap='gray')
plt.axis('off')

plt.subplot(1, 6, 3)
plt.imshow(input_image_np, cmap='gray') # add the origin parameter
plt.triplot(pos[:,1], pos[:,0], tri.simplices)
plt.axis('off')

plt.subplot(1, 6, 4)
plt.imshow(density_map.squeeze(0).squeeze(0).cpu().numpy(), cmap='hot', interpolation='nearest') # add the origin parameter
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
plt.figure(figsize=(10, 5))
plt.imshow(input_image_np, cmap='gray') # add the origin parameter
plt.triplot(pos[:,1], pos[:,0], tri.simplices)
plt.axis('off')
plt.show()

torch.cuda.empty_cache()

# %%
