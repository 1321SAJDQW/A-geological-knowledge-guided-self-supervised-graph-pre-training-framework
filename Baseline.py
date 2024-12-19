#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle as pkl
from torch_geometric.nn import GCNConv
from Loss import Contrastive_Loss
import argparse
import scipy.sparse as sp
import networkx as nx

# -------------------------------------------------------------------------------------------------------------
# Parameter
# -------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Graph-based Mineral Resource Prediction")

parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_size', type=int, default=256, help='Hidden layer size')
parser.add_argument('--emb_size', type=int, default=512, help='Embedding size')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--mask_rate_geo', type=float, default=0.1, help='Mask rate for geochemical features')
parser.add_argument('--l2', type=float, default=0.0005, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--lr_step_rate', type=float, default=0.85, help='Learning rate step rate.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--tau', type=float, default=5, help='Temperature in the contrastive loss.')
parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples for contrastive loss.')
parser.add_argument('--normalize', action='store_true', help='If to normalize the embedding.')

args = parser.parse_args()


np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------------------------------------------------------------------------------------------
# Load data
#-----------------------------------------------------------------------------------------------------------------------
with open('graph_750.pkl', 'rb') as rf1:
# with open('graph_500.pkl', 'rb') as rf1:
    graph = pkl.load(rf1)

data = pd.read_csv('HT_label.csv')
geo_features = data.drop(columns=['FID', 'POINT_X', 'POINT_Y', 'label','fault'])
geo_features = torch.from_numpy(geo_features.values).float()


if args.normalize:
    geo_features = F.normalize(geo_features)

labels = torch.tensor(data['label'].values, dtype=torch.long)

geo_features = geo_features.to(device)
labels = labels.to(device)

print(f"Geo Feature Dim: {geo_features.shape[1]}")

# -------------------------------------------------------------------------------------------------------------
# Function to mask features
def mask_features(features, mask_rate):
    mask = torch.rand(features.shape) < mask_rate
    masked_features = features.clone()
    masked_features[mask] = 0
    return masked_features

masked_geo_features = mask_features(geo_features, mask_rate=args.mask_rate_geo)

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    coo_tensor = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

    return coo_tensor.to_sparse_csr()

def adj_calculate(graph):
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = sp.coo_matrix(adj)
    adj_ori = torch.Tensor(adj.toarray()).to(device)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    adj_label = adj_ori + torch.eye(adj_ori.shape[0]).to(device)
    pos_weight = (adj_ori.shape[0] ** 2 - adj_ori.sum()) / adj_ori.sum()
    norm = adj_ori.shape[0] ** 2 / (2 * (adj_ori.shape[0] ** 2 - adj_ori.sum()))
    return adj, adj_ori, adj_label, pos_weight, norm

adj, adj_ori, adj_label, pos_weight, norm = adj_calculate(graph)


class SimpleGCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCNEncoder, self).__init__()
        self.gc1 = GCNConv(input_dim, hidden_dim)
        self.gc2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gc2(x, edge_index)
        return x

class Linear_Classifier(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(Linear_Classifier, self).__init__()
        self.fc1 = nn.Linear(ft_in, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, nb_classes)


    def forward(self, seq):
        #seq = F.relu(seq)
        ret = F.relu(self.fc1(seq))
        ret = F.relu(self.fc2(ret))
        ret = self.fc3(ret)
        return F.log_softmax(ret,dim=1)

GCN = SimpleGCNEncoder(masked_geo_features.shape[1], args.hidden_size, args.emb_size).to(device)


Classifier = Linear_Classifier(args.emb_size, labels.max().item()).to(device)

optimizer_encoder = torch.optim.Adam(GCN.parameters(), lr=args.lr, weight_decay=args.l2)
optimizer_classifier = torch.optim.Adam(Classifier.parameters(), lr=0.0001, weight_decay=0.0005)
scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=10, gamma=args.lr_step_rate)


for epoch in range(args.epochs):

    GCN.train()

    emb_before = GCN(geo_features, adj)
    emb_after = GCN(masked_geo_features, adj)

    Contrastive_loss = Contrastive_Loss(emb_before, emb_after, tau=args.tau)

    total_loss = Contrastive_loss

    optimizer_encoder.zero_grad()

    total_loss.backward()

    optimizer_encoder.step()

    scheduler_encoder.step()

    torch.cuda.empty_cache()

    print(f'Epoch {epoch}: Total_loss = {total_loss.item():.4f}')


GCN.eval()
emb = GCN(geo_features, adj).detach()

#-----------------------------------------------------------------------------------------------------------------------
# Train the classifier
#-----------------------------------------------------------------------------------------------------------------------
label1 = labels.to("cpu").numpy()
positiveIndex = np.where(label1 == 1)[0]
negativeIndex = np.where(label1 == 0)[0]
trainPositiveIndex = np.random.choice(positiveIndex, int(0.8 * positiveIndex.shape[0]), replace=False)
testPositiveIndex = np.setdiff1d(positiveIndex, trainPositiveIndex)
trainNegativeIndex = np.random.choice(negativeIndex, int(0.8 * negativeIndex.shape[0]), replace=False)
testNegativeIndex = np.setdiff1d(negativeIndex, trainNegativeIndex)
trainIndex = np.append(trainPositiveIndex, trainNegativeIndex)
testIndex = np.append(testPositiveIndex, testNegativeIndex)


for i in range(1000):
    optimizer_classifier.zero_grad()
    y_pred = Classifier(emb)
    classifier_loss = F.nll_loss(y_pred[trainIndex], labels[trainIndex])
    classifier_loss.backward()
    optimizer_classifier.step()
    print(f'Epoch {i}: Loss = {classifier_loss.item():.4f}')


#-----------------------------------------------------------------------------------------------------------------------
# Predictions
# ----------------------------------------------------------------------------------------------------------------------
torch.save(Classifier.state_dict(), "classifier_best.pth")
torch.save(GCN.state_dict(), "encoder_best.pth")

data = pd.read_csv('HT_label.csv')
emb = GCN(geo_features, adj)
out = Classifier(emb.detach())


probability = nn.functional.softmax(out, dim=-1)
result = probability[:, 1].to("cpu").detach().numpy()


data['Prediction_Probability'] = result
data.to_csv('Prediction_Baseline.csv', index=False)

