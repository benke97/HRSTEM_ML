#%%
import torch
import pandas as pd
import pickle
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DynamicEdgeConv, global_mean_pool
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import knn_graph
import numpy as np

print(torch.version.cuda)
with open('dataset_hist.pkl', 'rb') as f:
    data = pickle.load(f)
point_sets = data['dataframes']

data_geometric = []
for idx,df in enumerate(point_sets):
    x = torch.tensor(df[['x', 'y']].values, dtype=torch.float)
    y = torch.tensor(df['label'].values.astype(int), dtype=torch.float)
    data = Data(x=x, y=y)
    data_geometric.append(data)
print(len(data_geometric))
# Let's say you want to use 80% of the data for training and 20% for validation
split_index = int(len(data_geometric) * 0.8)

train_dataset = data_geometric[:split_index]
val_dataset = data_geometric[split_index:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.bn1 = torch.nn.BatchNorm1d(64) 
        self.bn2 = torch.nn.BatchNorm1d(64) 
        self.bn3 = torch.nn.BatchNorm1d(128) 
        self.bn4 = torch.nn.BatchNorm1d(256)

        self.conv1 = DynamicEdgeConv(MLP([2 * 2, 64]), k=20, aggr='max')
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64]), k=20, aggr='max')
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 128]), k=20, aggr='max')
        self.conv4 = DynamicEdgeConv(MLP([2 * 128, 256]), k=20, aggr='max')

        self.bn5 = torch.nn.BatchNorm1d(512) 
        self.bn6 = torch.nn.BatchNorm1d(256)

        self.lin1 = Linear(64 + 64 + 128 + 256, 512)
        self.lin2 = Linear(512, 256)
        self.lin3 = Linear(256, 1)

        self.dp1 = torch.nn.Dropout(0.5)
        self.dp2 = torch.nn.Dropout(0.5)

    def forward(self, data):
        x = data.x
        x = (x - x.mean(dim=0)) / x.std(dim=0)
        #print(f"x initial shape: {x.shape}, max: {x.max()}, min: {x.min()}, mean: {x.mean()}")

        x1 = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        #print(f"x1 shape: {x1.shape}, max: {x1.max()}, min: {x1.min()}, mean: {x1.mean()}")

        x2 = F.leaky_relu(self.bn2(self.conv2(x1)), negative_slope=0.2)
        #print(f"x2 shape: {x2.shape}, max: {x2.max()}, min: {x2.min()}, mean: {x2.mean()}")

        x3 = F.leaky_relu(self.bn3(self.conv3(x2)), negative_slope=0.2)
        #print(f"x3 shape: {x3.shape}, max: {x3.max()}, min: {x3.min()}, mean: {x3.mean()}")

        x4 = F.leaky_relu(self.bn4(self.conv4(x3)), negative_slope=0.2)
        #print(f"x4 shape: {x4.shape}, max: {x4.max()}, min: {x4.min()}, mean: {x4.mean()}")


        x = torch.cat((x1,x2,x3,x4),dim=-1)

        x = F.leaky_relu(self.bn5(self.lin1(x)), negative_slope=0.2)
        x = self.dp1(x)
        #print(f"x after lin1 shape: {x.shape}, max: {x.max()}, min: {x.min()}, mean: {x.mean()}")

        x = F.leaky_relu(self.bn6(self.lin2(x)), negative_slope=0.2)
        x = self.dp2(x)
        #print(f"x after lin2 shape: {x.shape}, max: {x.max()}, min: {x.min()}, mean: {x.mean()}")

        x = torch.sigmoid(self.lin3(x)).squeeze(-1)
        #print(f"x after lin3 shape: {x.shape}, max: {x.max()}, min: {x.min()}, mean: {x.mean()}")
        return x


class MLP(torch.nn.Module):
    def __init__(self, channels):
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(1, len(channels)):
            self.layers.append(Linear(channels[i - 1], channels[i]))

    def forward(self, x):
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x, inplace=True)
        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
#%%    
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = torch.nn.BCELoss()

def train(epoch):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training batches (Epoch {epoch+1})"):
        out = model(batch.to(device))
        loss = criterion(out,batch.y.to(device))
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Train Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    return avg_loss  # Add this line

def validate(epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch.to(device))
            loss = criterion(out,batch.y.to(device))
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    print(f"Val Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    return avg_loss  # Add this line

best_val_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(300):  # Adjust the number of epochs as needed
    train_loss = train(epoch)
    val_loss = validate(epoch)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        torch.save(model.state_dict(), 'best_model_weights.pt')
        model.load_state_dict(torch.load('best_model_weights.pt'))

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
#%%
# Let's say you want to visualize the first point set in the validation dataset
data = val_dataset[0]
model.eval()
with torch.no_grad():
    out = model(data.to(device))
predicted_labels = out.argmax(dim=1).numpy()

plt.scatter(data.pos[:, 0], data.pos[:, 1], c=predicted_labels)
plt.show()
# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
# %%
