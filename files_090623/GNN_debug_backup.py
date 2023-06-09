#%%
import torch
from torch_geometric.nn import MessagePassing
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
        self.conv5 = EdgeConv(64, num_edge_features, 32)
        self.batch_norm = torch.nn.BatchNorm1d(32)
        self.classifier = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv4(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv5(x, edge_index, edge_attr)
        x = self.batch_norm(x)
        x = F.dropout(x, training=self.training)

        x = self.classifier(x)

        return F.log_softmax(x, dim=1)
#%%    
with open('dataset_hist.pkl', 'rb') as f:
    data = pickle.load(f)

point_sets = data['dataframes']
print(len(point_sets))

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
    
    return np.array([length, cosine, sine])

def calculate_angle(p1, p2, p3):
    """
    Calculates the angle between three points.
    """
    v1 = p1 - p2
    v2 = p3 - p2

    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1, 1)  # ensure the value is within [-1, 1]
    return np.arccos(cos_theta)

data_list = []
for idx, df in enumerate(point_sets):  
    points = df[['x', 'y']].values
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
    
    edge_attr = []
    for edge in edge_index.t().numpy():
        edge_attr.append(edge_features(points, edge))
    edge_attr = np.array(edge_attr)

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    degree = np.sum(adjacency_matrix, axis=1)
    df['degree'] = degree
    
    df['x'] = (df['x'] - df['x'].min()) / (df['x'].max() - df['x'].min())
    df['y'] = (df['y'] - df['y'].min()) / (df['y'].max() - df['y'].min())
    
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

    #print(df['average_neighbor_distance'])
    #print("degree",df['degree'])
    node_features = torch.tensor(df[['relative_x', 'relative_y', 'degree','average_neighbor_distance','min_n_dist','max_n_dist', 'min_relative_angle', 'max_relative_angle']].values, dtype=torch.float)

    labels = torch.tensor(df['label'].values.astype(int), dtype=torch.long)
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=labels)

    data_list.append(data)

#%%
# Split your data into a training set and a validation set
train_size = int(0.75 * len(data_list))  # Assuming you want 80% of the data for training
val_size = len(data_list) - train_size
train_dataset, val_dataset = random_split(data_list, [train_size, val_size])

# Create a DataLoader for the training set and the validation set
train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define your model, loss function and optimizer
model = GCN(num_node_features=8, num_edge_features=3, num_classes=2)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Keep track of losses for each epoch
train_losses = []
val_losses = []

# Training function
def train(epoch):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training batches (Epoch {epoch+1})"):
        out = model(batch)
        loss = loss_fn(out, batch.y)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)  # Append to list
    print(f"Train Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Validation function
def validate(epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch)
            loss = loss_fn(out, batch.y)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    val_losses.append(avg_loss)  # Append to list
    print(f"Val Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Initialize best validation loss with a high value
best_val_loss = float('inf')

# Training and validation function
for epoch in range(300):  # Number of epochs
    train(epoch)
    validate(epoch)
    
    # Save the model weights at the best epoch
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        torch.save(model.state_dict(), 'best_model_weights.pt')

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
from torch_geometric.data import Batch
# Select the point_set you want to plot
point_set_idx = 94 # or whichever point_set you want to plot
point_set = val_dataset[point_set_idx]

# Convert point_set to a batch for model input
point_set = point_set.to('cpu')  # Assuming you are using CPU for inference
point_set = Batch.from_data_list([point_set])

# Run the point_set through the model to get the predicted labels
model.eval()  # Ensure model is in evaluation mode
with torch.no_grad():
    out = model(point_set)
predicted_labels = torch.argmax(out, dim=1).numpy()

# Extract the points and plot
points = point_set.x.numpy()
plt.scatter(points[:, 0], points[:, 1], c=predicted_labels)
plt.title(f'Point Set {point_set_idx} Predicted Labels')
plt.axis("off")
plt.axis("equal")
plt.show()
# %%
