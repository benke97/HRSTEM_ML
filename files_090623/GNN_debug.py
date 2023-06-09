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
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_attr has shape [E, edge_attr_dim]

        # Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Add edge attributes for self-loops.
        loop_attr = torch.zeros((x.size(0), edge_attr.size(1)), device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        # edge_attr has shape [E, edge_attr_dim, 1]
        #print(f'x_i shape (in model): {x_i.shape}')
        #print(f'x_j shape (in model): {x_j.shape}')
        #print(f'edge_attr shape (in model): {edge_attr.shape}')
        edge_attr = edge_attr.squeeze(-1)  # Remove the singleton dimension
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=1)  # shape [E, in_channels * 2 + edge_attr_dim]
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

def augment_pointset(points, rotation_angle, translation_vector):
    # First, calculate the center of the point set
    center = np.mean(points, axis=0)

    # Center the points around the origin
    points -= center

    # Define the rotation matrix
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                [np.sin(rotation_angle), np.cos(rotation_angle)]])

    # Rotate the points
    points = np.dot(points, rotation_matrix)

    # Translate the points
    points += translation_vector + center

    return points


def edge_features(points, edge):
    # Edge length
    length = np.linalg.norm(points[edge[0]] - points[edge[1]])
    
    if length == 0:
        print(f"Zero length encountered. Points: {points[edge[0]], points[edge[1]]}")
    
    # Calculate the edge vector, taking the point with the largest y value minus the other
    if points[edge[0], 1] > points[edge[1], 1]:
        vec = points[edge[0]] - points[edge[1]]
    else:
        vec = points[edge[1]] - points[edge[0]]
        
    # Cosine of the angle
    cosine = vec[0] / length
    # Sine of the angle
    sine = vec[1] / length
    
    return np.array([cosine,sine])

def calculate_angle(p1, p2, p3):
    """
    Calculates the angle between three points.
    """
    v1 = p1 - p2
    v2 = p3 - p2

    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1, 1)  # ensure the value is within [-1, 1]
    return np.arccos(cos_theta)

def graph_preprocess(idx,df):
    # Randomly rotate and translate the points
    angle = np.random.uniform(0, 2*np.pi) # rotation angle
    
    displacement = np.random.uniform(-128,128,2) # 2D displacement vector

    points = augment_pointset(df[['x', 'y']].values,angle,displacement)

    df['x'] = points[:, 0]
    df['y'] = points[:, 1]

    max_val = max(df['x'].max(), df['y'].max())

    df['x'] = (df['x']-df['x'].min())/max_val
    df['y'] = (df['y']-df['y'].min())/max_val
    points = df[['x', 'y']].values
    #print(points)
    tri = Delaunay(points)

    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
            edges.add(edge)

    adjacency_matrix = np.zeros((len(points), len(points)))
    
    num_nodes = len(points)
    print(f"Number of nodes in dataframe {idx}: {num_nodes}")

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    
    num_edges = edge_index.shape[1]
    print(f"Number of edges in dataframe {idx}: {num_edges}")


    for edge in edge_index.T:
        #print(f"Edge: {edge}, Edge[0]: {edge[0]}, Edge[1]: {edge[1]}")
        # Since it's an undirected graph, we add 1 for both (i, j) and (j, i)
        adjacency_matrix[edge[0], edge[1]] = 1
        adjacency_matrix[edge[1], edge[0]] = 1
    
    edge_attr_before_norm = []
    for edge in edge_index.t().numpy():
        edge_attr_before_norm.append(edge_features(points, edge))
    edge_attr_before_norm = np.array(edge_attr_before_norm)
    
    
    scaler_edge = StandardScaler()
    edge_attr = scaler_edge.fit_transform(edge_attr_before_norm)

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    degree = np.sum(adjacency_matrix, axis=1)
    df['degree'] = degree
    
    centroid_x = df['x'].mean()
    centroid_y = df['y'].mean()

    df['relative_x'] = df['x'] - centroid_x
    df['relative_y'] = df['y'] - centroid_y

    # Initialize average_neighbor_distance
    df['average_neighbor_distance'] = 0

    # Iterate over edges
    for edge in edge_index.T:
        # Compute distance between nodes
        distance = np.linalg.norm(points[edge[0]] - points[edge[1]])

        # Convert tensor to numpy for indexing
        edge_numpy = edge.numpy()

        # Add the distance to both nodes, we will average it later
        df.loc[edge_numpy[0], 'average_neighbor_distance'] += distance
        df.loc[edge_numpy[1], 'average_neighbor_distance'] += distance

    # Get the degrees of the nodes
    degrees = np.sum(adjacency_matrix, axis=1)

    # Average the distances
    df['average_neighbor_distance'] /= degrees

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



    #print(df['average_neighbor_distance'])
    #print("degree",df['degree'])
    node_features = torch.tensor(df[['relative_x','relative_y','degree', 'ratio_min_avg_dist', 'ratio_max_avg_dist', 'min_relative_angle', 'max_relative_angle', 'avg_diff_relative_angle']].values, dtype=torch.float)

    labels = torch.tensor(df['label'].values.astype(int), dtype=torch.long)
    data = Data(x=node_features, edge_index=edge_index,edge_attr=edge_attr, y=labels)

    return data
#%%    
with open('new_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

point_sets = data['dataframes']
print(len(point_sets))


#%%
data_list = []
for idx, df in enumerate(point_sets): 
    data = graph_preprocess(idx,df)
    data_list.append(data)

with open('data_list.pkl', 'wb') as f:
    pickle.dump(data_list, f)

#%%
# Split your data into a training set and a validation set
with open('data_list.pkl', 'rb') as f:
    data_list = pickle.load(f)
train_size = int(0.8 * len(data_list))  # Assuming you want 80% of the data for training
val_size = len(data_list) - train_size
train_dataset, val_dataset = random_split(data_list, [train_size, val_size])

# Create a DataLoader for the training set and the validation set
train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)
def custom_loss(output, target, edge_index):
    # Compute the standard Cross Entropy loss
    ce_loss = F.nll_loss(output, target)

    # Add a penalty term for misclassified nodes
    penalty = 0
    for node in range(output.size(0)):  # For each node in the graph
        neighbors = edge_index[1, edge_index[0] == node]  # Get the neighbors of the node
        # If there are no neighbors, continue to the next iteration
        if neighbors.numel() == 0:
            continue
        # If the node is predicted to be in one class but all its neighbors are in the other class
        if (output[node].argmax() != output[neighbors].argmax()).all():
            penalty += 0.1

    # Return the total loss
    return ce_loss + penalty
# Define your model, loss function and optimizer
model = GCN(num_node_features=8,num_edge_features=2, num_classes=2)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0005)

# Keep track of losses for each epoch
train_losses = []
val_losses = []

# Training function
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
    # Step the scheduler
    #scheduler.step()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)  # Append to list
    if epoch % 3 == 0:  # Print every 10 epochs
        print(f"Train Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Validation function
def validate(epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch)
            loss = loss_fn(out,batch.y)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    val_losses.append(avg_loss)  # Append to list
    if epoch % 3 == 0:  # Print every 10 epochs
        print(f"Val Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Initialize best validation loss with a high value
best_val_loss = float('inf')

# Training and validation function
for epoch in range(100):  # Number of epochs
    train(epoch)
    validate(epoch)
    
    
    # Save the model weights at the best epoch
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        torch.save(model.state_dict(), 'best_model_weights.pt')
        model.load_state_dict(torch.load('best_model_weights.pt'))

# Load the best model weights
model.load_state_dict(torch.load('best_model_weights.pt'))

# Plotting losses
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%
# Select the point_set you want to plot
def graph_preprocess_2(df):
    angle = np.random.uniform(0, 2*np.pi) # rotation angle
    
    displacement = np.random.uniform(-128,128,2) # 2D displacement vector

    points = augment_pointset(df[['x', 'y']].values,angle,displacement)

    df['x'] = points[:, 0]
    df['y'] = points[:, 1]

    max_val = max(df['x'].max(), df['y'].max())

    df['x'] = df['x']/max_val
    df['y'] = df['y']/max_val

    points = df[['x', 'y']].values
    #print(points)
    tri = Delaunay(points)

    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
            edges.add(edge)

    adjacency_matrix = np.zeros((len(points), len(points)))
    
    num_nodes = len(points)
    #print(f"Number of nodes in dataframe {idx}: {num_nodes}")

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    
    num_edges = edge_index.shape[1]
    #print(f"Number of edges in dataframe {idx}: {num_edges}")


    for edge in edge_index.T:
        #print(f"Edge: {edge}, Edge[0]: {edge[0]}, Edge[1]: {edge[1]}")
        # Since it's an undirected graph, we add 1 for both (i, j) and (j, i)
        adjacency_matrix[edge[0], edge[1]] = 1
        adjacency_matrix[edge[1], edge[0]] = 1
    
    edge_attr_before_norm = []
    for edge in edge_index.t().numpy():
        edge_attr_before_norm.append(edge_features(points, edge))
    edge_attr_before_norm = np.array(edge_attr_before_norm)
    
    
    scaler_edge = StandardScaler()
    edge_attr = scaler_edge.fit_transform(edge_attr_before_norm)

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    degree = np.sum(adjacency_matrix, axis=1)
    df['degree'] = degree
    
    centroid_x = df['x'].mean()
    centroid_y = df['y'].mean()

    df['relative_x'] = df['x'] - centroid_x
    df['relative_y'] = df['y'] - centroid_y

    # Initialize average_neighbor_distance
    df['average_neighbor_distance'] = 0

    # Iterate over edges
    for edge in edge_index.T:
        # Compute distance between nodes
        distance = np.linalg.norm(points[edge[0]] - points[edge[1]])

        # Convert tensor to numpy for indexing
        edge_numpy = edge.numpy()

        # Add the distance to both nodes, we will average it later
        df.loc[edge_numpy[0], 'average_neighbor_distance'] += distance
        df.loc[edge_numpy[1], 'average_neighbor_distance'] += distance

    # Get the degrees of the nodes
    degrees = np.sum(adjacency_matrix, axis=1)

    # Average the distances
    df['average_neighbor_distance'] /= degrees

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



    #print(df['average_neighbor_distance'])
    #print("degree",df['degree'])
    node_features = torch.tensor(df[['relative_x','relative_y','degree', 'ratio_min_avg_dist', 'ratio_max_avg_dist', 'min_relative_angle', 'max_relative_angle', 'avg_diff_relative_angle']].values, dtype=torch.float)


    data = Data(x=node_features, edge_index=edge_index,edge_attr=edge_attr)

    return data,df





with open('new_dataset.pkl', 'rb') as f:
    data = pickle.load(f)
model = GCN(num_node_features=8,num_edge_features=2, num_classes=2)
point_sets = data['dataframes']
model.load_state_dict(torch.load('best_model_weights.pt'))
point_set_idx = 1337 # or whichever point_set you want to plot
point_set = point_sets[point_set_idx]
points = point_set[['x','y']].values
plt.figure()
plt.scatter(points[:,0],points[:,1])
plt.show()
print(points)
point_set,df = graph_preprocess_2(point_set)

# Convert point_set to a batch for model input
point_set = point_set.to('cpu')  # Assuming you are using CPU for inference
#point_set = Batch.from_data_list([point_set])

# Run the point_set through the model to get the predicted labels
model.eval()  # Ensure model is in evaluation mode
with torch.no_grad():
    out = model(point_set)
predicted_labels = torch.argmax(out, dim=1).numpy()

# Extract the points and plot
point_sets = data['dataframes']
point_set = point_sets[point_set_idx]
points = point_set[['x','y']].values
#print(points)

plt.scatter(points[:, 0], points[:, 1], c=predicted_labels)
plt.title(f'Point Set {point_set_idx} Predicted Labels')
#plt.axis("off")
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
gnn = GCN(num_node_features=8,num_edge_features=2, num_classes=2)
gnn.load_state_dict(torch.load('best_model_weights.pt'))

unet = UNet()
unet = unet.to(device)
loaded_data = torch.load("model_data_epoch_44.pth")
loaded_model_state_dict = loaded_data['model_state_dict']
unet.load_state_dict(loaded_model_state_dict)

with open('new_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

point_sets = data['dataframes']

result_images = []
for image_path in glob.glob(os.path.join(image_directory_path, "*.tif")):

    pos_pre =  point_sets[10][['x', 'y']].values
    point_sett,dff = graph_preprocess_2(point_sets[10])
    pos_post = point_sett.x.numpy()

    print(image_path)
    raw_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    unet.eval()
    with torch.no_grad():
        predicted_output = unet(image_tensor)
        predicted_segmentation = postprocess_output(predicted_output)
    pred_positions = analyzer.return_positions_experimental(raw_image,predicted_segmentation)

    point_set_exp = pd.DataFrame({
        "x":pred_positions[:,0]/128,
        "y":pred_positions[:,1]/128
    })
    point_setta,dff = graph_preprocess_2(point_set_exp)
    points = point_setta.x.numpy()
    #plt.figure()
    #plt.scatter(points[:, 0], points[:, 1], c='r')
    #plt.scatter(pred_positions[:, 1]/128, pred_positions[:, 0]/128,c=predicted_labels)

    point_set = point_setta.to('cpu')
    gnn.eval()  # Ensure model is in evaluation mode
    with torch.no_grad():
        out = gnn(point_setta)
    #print(out)
    predicted_labels = torch.argmax(out, dim=1).numpy()
    #print(predicted_labels)
    #plt.axis("equal")
    
    plt.figure()
    #plt.scatter(pos_pre[:, 0], pos_pre[:, 1], c='b')
    plt.scatter(pos_post[:, 1], pos_post[:, 0],c='r')
    plt.axis("equal")
    plt.title('xd')

    #plt.axis("equal")


    plt.figure()
    #plt.scatter(points[:, 0], points[:, 1], c='r')
    #plt.imshow(raw_image,cmap="gray")
    plt.scatter(pred_positions[:, 1],pred_positions[:, 0],c=predicted_labels)
    #plt.xlim(0,1)
    #plt.ylim(0,1)

    experimental_graph, experimental_df = graph_preprocess_2(point_set_exp)
    simulated_graph, simulated_df = graph_preprocess_2(point_sets[6])    
    features = ['x','y','degree', 'ratio_min_avg_dist', 'ratio_max_avg_dist', 'min_relative_angle', 'max_relative_angle', 'avg_diff_relative_angle']

    #for feature in features:
    #    plt.figure(figsize=(10,5))
    #    plt.hist(experimental_df[feature], bins=50, alpha=0.5, label='Experimental')
    #    plt.hist(simulated_df[feature], bins=50, alpha=0.5, label='Simulated')
    #    plt.title(f"Distribution of {feature}")
    #    plt.xlabel(feature)
    #    plt.ylabel("Count")
    #    plt.legend()
    #    plt.show()
# %%