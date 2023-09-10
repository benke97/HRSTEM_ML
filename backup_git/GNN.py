#%%
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F
from torch_geometric.data import Data
import pickle
import pandas as pd
from scipy.spatial import Delaunay
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch.optim as optim

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='add') # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels * 2, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_attr has shape [E, edge_attr_dim]

        # Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        # edge_attr has shape [E, edge_attr_dim, 1]
        print(f'x_i shape (in model): {x_i.shape}')
        print(f'x_j shape (in model): {x_j.shape}')
        print(f'edge_attr shape (in model): {edge_attr.shape}')
        edge_attr = edge_attr.squeeze(-1)  # Remove the singleton dimension
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=1)  # shape [E, in_channels * 2 + edge_attr_dim]
        return self.lin(edge_input)

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = EdgeConv(num_node_features + num_edge_features, 128)
        self.conv2 = EdgeConv(128 + num_edge_features, 64)
        self.conv3 = EdgeConv(64 + num_edge_features, 32)
        self.classifier = torch.nn.Linear(32, num_classes)
        print(f"in_features of linear layer: {self.conv1.lin.in_features}")

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv3(x, edge_index, edge_attr)
        x = F.dropout(x, training=self.training)

        x = self.classifier(x)

        return F.log_softmax(x, dim=1)
    

with open('dataset_hist.pkl', 'rb') as f:
    data = pickle.load(f)

point_sets = data['dataframes']
print(len(point_sets))

data_list = []  # This will store your graph data

for idx, df in enumerate(point_sets):  # Assuming dataframes is your list of DataFrames
    # Create node features tensor from 'x' and 'y' columns
    # Compute Delaunay triangulation and get adjacency matrix
    tri = Delaunay(df[['x', 'y']].values)
    adjacency_matrix = np.zeros((len(df), len(df)))
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(3):
                adjacency_matrix[simplex[i], simplex[j]] = 1
    

    degree = np.sum(adjacency_matrix, axis=1)
    #print(adjacency_matrix)
    
    df['degree'] = degree
    
    df['x'] = (df['x'] - df['x'].min()) / (df['x'].max() - df['x'].min())
    df['y'] = (df['y'] - df['y'].min()) / (df['y'].max() - df['y'].min())
    
    centroid_x = df['x'].mean()
    centroid_y = df['y'].mean()

    df['relative_x'] = df['x'] - centroid_x
    df['relative_y'] = df['y'] - centroid_y
    
    edges = np.vstack(np.triu_indices_from(adjacency_matrix, k=1))
    df_edges = pd.DataFrame({'node_1': edges[0], 'node_2': edges[1]})

    df_edges = pd.DataFrame({'node_1': edges[0], 'node_2': edges[1]})
    condition = df.loc[df_edges['node_2'], 'relative_y'].values >= df.loc[df_edges['node_1'], 'relative_y'].values
    df_edges['delta_x'] = np.where(condition, 
                                   df.loc[df_edges['node_2'], 'relative_x'].values - df.loc[df_edges['node_1'], 'relative_x'].values,
                                   df.loc[df_edges['node_1'], 'relative_x'].values - df.loc[df_edges['node_2'], 'relative_x'].values)
    df_edges['delta_y'] = np.where(condition,
                                   df.loc[df_edges['node_2'], 'relative_y'].values - df.loc[df_edges['node_1'], 'relative_y'].values,
                                   df.loc[df_edges['node_1'], 'relative_y'].values - df.loc[df_edges['node_2'], 'relative_y'].values)

    df_edges['edge_length'] = np.sqrt(df_edges['delta_x']**2 + df_edges['delta_y']**2)
    angles = np.arctan2(df_edges['delta_y'], df_edges['delta_x'])
    
    df_edges['sin_angle'] = np.sin(angles)
    df_edges['cos_angle'] = np.cos(angles)

    edge_attr = torch.tensor(df_edges[['sin_angle', 'cos_angle', 'edge_length']].values, dtype=torch.float)
    #print(f'edge_attr shape (after generation): {edge_attr.shape}')

    node_features = torch.tensor(df[['relative_x', 'relative_y', 'degree']].values, dtype=torch.float)

    edge_index = torch.tensor(np.array(np.where(adjacency_matrix == 1)), dtype=torch.long)
    #print(f'edge_index shape (after generation): {edge_index.shape}')

    labels = torch.tensor(df['label'].values.astype(int), dtype=torch.long)
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=labels)

    data_list.append(data)


edge_index = torch.tensor([
    [0, 0, 1, 2, 3],
    [1, 2, 2, 3, 0]
], dtype=torch.long)

# Define node features (here we'll use 3 features per node)
x = torch.tensor([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 2.0],
    [1.0, 0.0, 3.0],
    [1.0, 1.0, 4.0]
], dtype=torch.float)

# Define edge features (we'll use 3 features per edge)
edge_attr = torch.tensor([
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0]
], dtype=torch.float)

# Define labels for each node
y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

# Create the graph data object
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

# Now you can pass this data object into your model to see if it works
model = GCN(num_node_features=3, num_edge_features=3, num_classes=2)
output = model(data)
print(output)
#%%
#print(f'data_list[0] details:\n {data_list[0]}')  # Print details of the first Data object in data_list
print("BOB")
print(data_list[0].num_node_features,data_list[0].edge_attr.size(1))
print("BOB")
model = GCN(num_node_features=data_list[0].num_node_features, num_edge_features=data_list[0].edge_attr.size(1), num_classes=2)
#print(f'model details: \n{model}')  # Print details of the model

# Move everything to the right device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Convert data_list to a DataLoader
loader = DataLoader(data_list, batch_size=32, shuffle=True)

# Train the model
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Print details of the model
print(f'model details: \n{model}')  

# Move everything to the right device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for data in data_list:
    data = data.to(device)

# Convert data_list to a DataLoader
loader = DataLoader(data_list, batch_size=1, shuffle=True)
print(f'edge_index shape (before model): {data.edge_index.shape}')
print(f'edge_attr shape (before model): {data.edge_attr.shape}')
# Train the model
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Get a single batch from your data loader
batch = next(iter(loader))

# Run the model with this batch and print the output shape
output = model(batch)
print(f'output shape: {output.shape}')

# Also, print the shapes of all model parameters
for name, param in model.named_parameters():
    print(f'{name}: {param.shape}')
#%%
for epoch in range(100):
    total_loss = 0
    for data in tqdm(loader, desc="Epoch: {}".format(epoch)):
        data = data.to(device)  # Move data to device
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("Loss: {:.4f}".format(total_loss / len(loader)))
#%%
# Test the model
model.eval()
correct = 0
total = 0
for data in loader:
    _, pred = model(data).max(dim=1)
    correct += float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    total += data.test_mask.sum().item()

accuracy = correct / total
print('Accuracy: {:.4f}'.format(accuracy))