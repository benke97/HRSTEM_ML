#%%
import torch
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from scipy.spatial import Delaunay
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch.optim as optim
import math
from tqdm import tqdm
from itertools import combinations
from torch_geometric.data import Batch
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EdgeConv(MessagePassing):
    def __init__(self, num_node_features, num_edge_features, out_channels):
        super(EdgeConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(num_node_features * 2 + num_edge_features, out_channels)

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        loop_attr = torch.zeros((x.size(0), edge_attr.size(1)), device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        edge_attr = edge_attr.squeeze(-1)
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.lin(edge_input)

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = EdgeConv(num_node_features, num_edge_features, 512)
        self.conv2 = EdgeConv(512, num_edge_features, 256)
        self.conv3 = EdgeConv(256, num_edge_features, 128)
        self.conv4 = EdgeConv(128, num_edge_features, 64)
        self.conv5 = EdgeConv(512+256+128+64, num_edge_features, 32)
        self.batch_norm = torch.nn.BatchNorm1d(32)
        self.classifier = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x1 = F.dropout(x, training=self.training)

        x = self.conv2(x1, edge_index, edge_attr)
        x = F.relu(x)
        x2 = F.dropout(x, training=self.training)
        
        x = self.conv3(x2, edge_index, edge_attr)
        x = F.relu(x)
        x3 = F.dropout(x, training=self.training)       
        
        x = self.conv4(x3, edge_index, edge_attr)
        x = F.relu(x)
        x4 = F.dropout(x, training=self.training)

        x = torch.cat((x1,x2,x3,x4),dim=1)

        x = self.conv5(x, edge_index, edge_attr)
        x = self.batch_norm(x)
        x = F.dropout(x, training=self.training)

        x = self.classifier(x)

        return F.log_softmax(x, dim=1)
    
def rotate_pointset(points):
    rotation_angle = np.random.uniform(0, 2*np.pi)
    center = np.mean(points, axis=0)

    points -= center

    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                [np.sin(rotation_angle), np.cos(rotation_angle)]])

    points = np.dot(points, rotation_matrix)

    points += center

    return points

def normalize_pointset(points):
    if points[:,0].min() < 0 or points[:,1].min() < 0:
        points -= [points[:,0].min(),points[:,1].min()]
    
    normalizing_factor = max(points[:,0].max(), points[:,1].max())
    points = points/normalizing_factor
    
    assert np.all((0 <= points) & (points <= 1))
    
    return points

def edge_features(points, edge):
    # Edge length
    length = np.linalg.norm(points[edge[0]] - points[edge[1]])
    
    if length == 0:
        print(f"Zero length encountered. Points: {points[edge[0]], points[edge[1]]}")
    
    vec = points[edge[0]] - points[edge[1]]

    cosine = vec[0] / length

    sine = vec[1] / length
    
    return np.array([length,0])

def calculate_angle(p1, p2, p3):

    v1 = p1 - p2
    v2 = p3 - p2

    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1, 1)  # ensure the value is within [-1, 1]
    return np.arccos(cos_theta)

def graph_preprocess(df,idx=0,experimental=False):
    # Randomly rotate and translate the points
    points = rotate_pointset(df[['x', 'y']].values)
    points = normalize_pointset(points)

    df['x'] = points[:, 0]
    df['y'] = points[:, 1]

    tri = Delaunay(points)

    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
            edges.add(edge)

    adjacency_matrix = np.zeros((len(points), len(points)))
    
    num_nodes = len(points)

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    
    num_edges = edge_index.shape[1]
    
    if idx != 0:
        print(f"Number of nodes in dataframe {idx}: {num_nodes}")
        print(f"Number of edges in dataframe {idx}: {num_edges}")


    for edge in edge_index.T:
        adjacency_matrix[edge[0], edge[1]] = 1
        adjacency_matrix[edge[1], edge[0]] = 1
    
    edge_attr = []
    for edge in edge_index.t().numpy():
        edge_attr.append(edge_features(points, edge))

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)


    df['average_neighbor_distance'] = 0
    for edge in edge_index.T:
        distance = np.linalg.norm(points[edge[0]] - points[edge[1]])

        edge_numpy = edge.numpy()

        df.loc[edge_numpy[0], 'average_neighbor_distance'] += distance
        df.loc[edge_numpy[1], 'average_neighbor_distance'] += distance

    degrees = np.sum(adjacency_matrix, axis=1)
    df['average_neighbor_distance'] /= degrees

    degree = np.sum(adjacency_matrix, axis=1)
    df['degree'] = degree / np.max(degree)


    df['min_n_dist'] = 0
    df['max_n_dist'] = 0
    # Compute pairwise distance matrix
    distance_matrix = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)

    for i in range(len(points)):
        # Get indices of neighbors
        neighbors = np.where(adjacency_matrix[i, :] > 0)[0]

        if len(neighbors) > 0:  # Check if node has any neighbors
            # Get distances to neighbors
            neighbor_distances = distance_matrix[i, neighbors]

            # Compute minimum and maximum neighbor distance
            min_n_dist = np.min(neighbor_distances)
            max_n_dist = np.max(neighbor_distances)

            # Update the dataframe
            df.loc[i, 'min_n_dist'] = min_n_dist
            df.loc[i, 'max_n_dist'] = max_n_dist
        else:  # For isolated nodes, we'll set the distance to inf
            df.loc[i, 'min_n_dist'] = np.inf
            df.loc[i, 'max_n_dist'] = np.inf


    df['min_relative_angle'] = 0
    df['max_relative_angle'] = 0
    
    for i in range(len(points)):
        # Get indices of neighbors
        neighbors = np.where(adjacency_matrix[i, :] > 0)[0]

        if len(neighbors) > 1:  # Check if node has more than one neighbor
            # Calculate all angles between neighbors
            angles = [calculate_angle(points[i], points[n1], points[n2]) for n1, n2 in combinations(neighbors, 2)]
            
            # Compute minimum and maximum neighbor angle
            min_angle = np.min(angles)
            max_angle = np.max(angles)

            # Update the dataframe
            df.loc[i, 'min_relative_angle'] = min_angle
            df.loc[i, 'max_relative_angle'] = max_angle
        else:  # For nodes with one or no neighbors, we'll set the angles to 0
            df.loc[i, 'min_relative_angle'] = 0
            df.loc[i, 'max_relative_angle'] = 0


    df['ratio_max_avg_dist'] = 0
    df['ratio_min_avg_dist'] = 0

    for i in range(len(points)):
        # Get indices of neighbors
        neighbors = np.where(adjacency_matrix[i, :] > 0)[0]

        if len(neighbors) > 0:  # Check if node has any neighbors
            # Get distances to neighbors
            neighbor_distances = distance_matrix[i, neighbors]

            # Compute minimum and maximum neighbor distance
            min_n_dist = np.min(neighbor_distances)
            max_n_dist = np.max(neighbor_distances)

            # Calculate ratios and Update the dataframe
            if df.loc[i, 'average_neighbor_distance'] != 0:
                df.loc[i, 'ratio_max_avg_dist'] = max_n_dist / df.loc[i, 'average_neighbor_distance']
                df.loc[i, 'ratio_min_avg_dist'] = min_n_dist / df.loc[i, 'average_neighbor_distance']
            else:
                df.loc[i, 'ratio_max_avg_dist'] = np.inf
                df.loc[i, 'ratio_min_avg_dist'] = np.inf

    df['avg_diff_relative_angle'] = 0
    
    for i in range(len(points)):
        # Get indices of neighbors
        neighbors = np.where(adjacency_matrix[i, :] > 0)[0]

        if len(neighbors) > 1:  # Check if node has more than one neighbor
            # Calculate all angles between neighbors
            angles = [calculate_angle(points[i], points[n1], points[n2]) for n1, n2 in combinations(neighbors, 2)]
            
            # Compute average difference from mean angle
            mean_angle = np.mean(angles)
            avg_diff_angle = np.mean(np.abs(angles - mean_angle))

            # Update the dataframe
            df.loc[i, 'avg_diff_relative_angle'] = avg_diff_angle
        else:  # For nodes with one or no neighbors, we'll set the avg_diff_relative_angle to 0
            df.loc[i, 'avg_diff_relative_angle'] = 0

    df['ratio_min_avg_dist'] /= df['ratio_min_avg_dist'].max()
    df['ratio_max_avg_dist'] /= df['ratio_max_avg_dist'].max()
    df['min_relative_angle'] /= df['min_relative_angle'].max()
    df['max_relative_angle'] /= df['max_relative_angle'].max()
    df['avg_diff_relative_angle'] /= df['avg_diff_relative_angle'].max()
    node_features = torch.tensor(df[['degree', 'ratio_min_avg_dist', 'ratio_max_avg_dist', 'min_relative_angle', 'max_relative_angle', 'avg_diff_relative_angle']].values, dtype=torch.float)
    assert (0 <= node_features).all() and (node_features <= 1).all(), "Not all features are in the range [0, 1]"
    if not experimental:
        labels = torch.tensor(df['label'].values.astype(int), dtype=torch.long)
        data = Data(x=node_features, edge_index=edge_index,edge_attr=edge_attr, y=labels)
        return data
    else:
        data = Data(x=node_features, edge_index=edge_index,edge_attr=edge_attr)
        return data, df

#%% DATASET
with open('new_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

point_sets = data['dataframes']

data_list = []
for idx, df in enumerate(point_sets): 
    data = graph_preprocess(df,idx)
    data_list.append(data)

with open('data_list_normalized.pkl', 'wb') as f:
    pickle.dump(data_list, f)
#%%
with open('data_list_normalized.pkl', 'rb') as f:
    data_list = pickle.load(f)

train_size = int(0.8 * len(data_list))
val_size = len(data_list) - train_size
train_dataset, val_dataset = random_split(data_list, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)
# %%
import matplotlib.pyplot as plt

def collect_features(loader, attr):
    all_features = []
    for data in loader:
        all_features.append(getattr(data, attr).numpy())
    return np.concatenate(all_features)

# Collect features
node_features_train = collect_features(train_loader, 'x')
edge_features_train = collect_features(train_loader, 'edge_attr')

node_features_val = collect_features(val_loader, 'x')
edge_features_val = collect_features(val_loader, 'edge_attr')

# Combine training and validation features
node_features = np.concatenate([node_features_train, node_features_val])
edge_features = np.concatenate([edge_features_train, edge_features_val])

# Compute histograms
n_node_features = node_features.shape[1]
n_edge_features = edge_features.shape[1]
bins = np.linspace(0, 1, 50)  # 50 bins between 0 and 1

# Create subplots
fig, axs = plt.subplots(n_node_features + n_edge_features, figsize=(10, (n_node_features + n_edge_features)*5))

# Plot node features
node_feature_names = ['degree', 'ratio_min_avg_dist', 'ratio_max_avg_dist', 'min_relative_angle', 'max_relative_angle', 'avg_diff_relative_angle']
for i in range(n_node_features):
    hist_values = np.histogram(node_features[:, i], bins=bins)
    avg_hist = hist_values[0] / len(node_features)
    axs[i].bar(bins[:-1], avg_hist, width=np.diff(bins))
    axs[i].set_title(f'Node Feature: {node_feature_names[i]}')

# Plot edge features
edge_feature_names = ['edge length','xd']  # Replace with actual names if available
for i in range(n_edge_features):
    hist_values = np.histogram(edge_features[:, i], bins=bins)
    avg_hist = hist_values[0] / len(edge_features)
    axs[n_node_features + i].bar(bins[:-1], avg_hist, width=np.diff(bins))
    axs[n_node_features + i].set_title(f'Edge Feature: {edge_feature_names[i]}')

plt.tight_layout()
plt.show()
# %%

model = GCN(num_node_features=6,num_edge_features=2, num_classes=2)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
val_losses = []


def train(epoch):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training batches (Epoch {epoch+1})"):
        out = model(batch)
        loss = loss_fn(out,batch.y)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    if epoch % 3 == 0:  
        print(f"Train Epoch {epoch+1}, Loss: {avg_loss:.4f}")

def validate(epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch)
            loss = loss_fn(out,batch.y)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    val_losses.append(avg_loss)
    if epoch % 3 == 0:
        print(f"Val Epoch {epoch+1}, Loss: {avg_loss:.4f}")

best_val_loss = float('inf')

for epoch in range(50):
    train(epoch)
    validate(epoch)
    
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        torch.save(model.state_dict(), 'best_model_weights.pt')
        model.load_state_dict(torch.load('best_model_weights.pt'))

model.load_state_dict(torch.load('best_model_weights.pt'))

plt.plot(range(1, len(train_losses)+1), train_losses, label='Train')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# %%
from Analyzer import Analyzer
from PI_U_Net import UNet
import cv2
import glob
import os

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

image_directory_path = "data/experimental_data/32bit/"

analyzer = Analyzer()
gnn = GCN(num_node_features=6,num_edge_features=2, num_classes=2)
gnn.load_state_dict(torch.load('best_model_weights.pt'))

unet = UNet()
unet = unet.to(device)
loaded_data = torch.load("model_data_epoch_44.pth")
loaded_model_state_dict = loaded_data['model_state_dict']
unet.load_state_dict(loaded_model_state_dict)

image_path = glob.glob(os.path.join(image_directory_path, "*.tif"))[11]
raw_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
image_tensor = preprocess_image(image_path)
image_tensor = image_tensor.to(device)

unet.eval()
with torch.no_grad():
    predicted_output = unet(image_tensor)
    predicted_segmentation = postprocess_output(predicted_output)
pred_positions = analyzer.return_positions_experimental(raw_image,predicted_segmentation)

point_set_exp = pd.DataFrame({
    "x":pred_positions[:,0],
    "y":pred_positions[:,1]
})

point_setta,dff = graph_preprocess(point_set_exp,idx=0,experimental=True)
points = point_setta.x.numpy()

point_set = point_setta.to('cpu')
gnn.eval() 
with torch.no_grad():
    out = gnn(point_setta)

predicted_labels = torch.argmax(out, dim=1).numpy()

plt.figure()
plt.scatter(pred_positions[:, 1],pred_positions[:, 0],c=predicted_labels)
plt.gca().invert_yaxis()
plt.axis("equal")
# %%
