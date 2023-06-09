#%%
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import VGAE
import torch.nn.functional as F
from itertools import combinations
from scipy.spatial import Delaunay
import numpy as np
from torch_geometric.data import Data
import pickle
from tqdm import tqdm

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class GAEModel(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GAEModel, self).__init__()
        self.encoder = Encoder(num_node_features, out_channels=16)
        self.decoder = VGAE(self.encoder)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = self.decoder.encode(x, edge_index)
        return self.decoder.recon_loss(z, edge_index), self.decoder.kl_loss()

def calculate_angle(p1, p2, p3):
    """
    Calculates the angle between three points.
    """
    v1 = p1 - p2
    v2 = p3 - p2

    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1, 1)  # ensure the value is within [-1, 1]
    return np.arccos(cos_theta)

#%%
def graph_preprocess(idx,df):
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
    print(f"Number of nodes in dataframe {idx}: {num_nodes}")

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    
    num_edges = edge_index.shape[1]
    print(f"Number of edges in dataframe {idx}: {num_edges}")


    for edge in edge_index.T:
        #print(f"Edge: {edge}, Edge[0]: {edge[0]}, Edge[1]: {edge[1]}")
        # Since it's an undirected graph, we add 1 for both (i, j) and (j, i)
        adjacency_matrix[edge[0], edge[1]] = 1
        adjacency_matrix[edge[1], edge[0]] = 1
    
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

    data = Data(x=node_features, edge_index=edge_index)

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

with open('data_list_GAE.pkl', 'wb') as f:
    pickle.dump(data_list, f)
#%%
with open('data_list_GAE.pkl', 'rb') as f:
    data_list = pickle.load(f)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAEModel(num_node_features=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    total_loss = 0
    for data in tqdm(data_list, desc='Training', unit='batch'):
        data = data.to(device)
        optimizer.zero_grad()
        recon_loss, kl_loss = model(data)
        loss = recon_loss + kl_loss
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return total_loss / len(data_list)

for epoch in range(100):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
# %%
