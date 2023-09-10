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
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

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
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) # 2D rotation matrix
    displacement = np.random.uniform(-1, 1, 2) # 2D displacement vector

    points = df[['x', 'y']].values
    points = np.dot(points, rotation_matrix) + displacement

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
    node_features_before_norm = df[['relative_x','relative_y','degree', 'ratio_min_avg_dist', 'ratio_max_avg_dist', 'min_relative_angle', 'max_relative_angle', 'avg_diff_relative_angle']].values
    # Initialize a StandardScaler object
    scaler = StandardScaler()

    # Fit and transform the node features to have mean 0 and unit variance
    node_features = scaler.fit_transform(node_features_before_norm)

    # Create a tensor for node features
    node_features = torch.tensor(node_features, dtype=torch.float)
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

with open('data_list_GAE.pkl', 'wb') as f:
    pickle.dump(data_list, f)

#%%
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

classifier = svm.SVC(kernel='linear')  # You can change the kernel based on your problem
classifier.fit(features_train, labels_train)
labels_pred = classifier.predict(features_test)
accuracy = accuracy_score(labels_test, labels_pred)
print(f'The accuracy of the SVM is: {accuracy*100}%')