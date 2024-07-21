from semi_vae import _sample
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import torch.nn as nn
import dgl.data
import scipy.sparse as sp
from tqdm import tqdm

BENIGN_LABEL = 0
ANOMALY_LABEL = 1

class MLP(nn.Module):
    def __init__(self, input_dim=139, num_layers=2):
        super(MLP, self).__init__()

        layers = []
        # in_dim = input_dim
        # hidden_dim = in_dim // 2
        # for _ in range(num_layers-1):
        #     layers.append(nn.Linear(in_dim, hidden_dim))
        #     layers.append(nn.ReLU())

        #     if hidden_dim <= 4:
        #         in_dim = 4
        #         hidden_dim = 4
        #     else:
        #         in_dim = hidden_dim
        #         hidden_dim = hidden_dim // 2
        layers.append(nn.Linear(input_dim, 2))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

        # self.model = nn.Sequential(
        #     nn.Linear(input_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 2),
        #     nn.Sigmoid()
        # )
    
    def forward(self, x):
        return self.model(x)

class MLP2(nn.Module):
    def __init__(self, input_dim=139, num_layers=2):
        super(MLP2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

def normalize(x):
    x_tens = torch.stack(x)
    x_sort = torch.sort(x_tens)
    val = x_sort.values
    return (val-val[0]) / (val[-1]-val[0]), x_sort.indices

def testing(test_dataset, autoencoder, classification_layer):
    pred = []
    labels = test_dataset.y
    print("### Testing the architecture ###")
    data = autoencoder.forward(test_dataset.x)
    data = _sample(data[1], data[2])
    ps = classification_layer.forward(data)
    for p in tqdm(ps):
        if p[1] > p[0]:
            pred.append(1)
        else:
            pred.append(0)

    score = roc_auc_score(labels.tolist(), pred)
    switch_score = roc_auc_score(switch(labels.tolist()), pred)
    print(f':clown: {score} and {switch_score} - {sum(pred)} vs {len(pred)}')
    return max(score, switch_score)

def switch(arr):
    arr = [0 if a == 1 else 1 for a in arr]
    return arr

def testing_graph_dagad(test_dataset, feats, labels, autoencoder, classification_layer):
    pred = []
    print("### Testing the architecture ###")
    features = torch.Tensor(sp.csr_matrix(feats.cpu()).toarray()).to('cuda')
    adj = gen_edge_index(test_dataset).to('cuda')
    encoded = autoencoder.encoder(features, adj, permute=False, return_mulog=False)
    ps = classification_layer.forward(encoded)
    for p in ps:
        if abs(p[1]-p[0]) > 0.1:
            hi = 9
        if p[1] > p[0]:
            pred.append(1)
        else:
            pred.append(0)

    score = roc_auc_score(labels.tolist(), pred)
    switch_score = roc_auc_score(switch(labels.tolist()), pred)
    print(f':clown: {score} and {switch_score} - {sum(pred)} vs {len(pred)}')
    return max(score, switch_score)

def testing_graph(test_dataset, feats, labels, autoencoder, classification_layer):
    pred = []
    print("### Testing the architecture ###")
    encoded = autoencoder.encoder(test_dataset, feats, return_mulog=False)
    ps = classification_layer.forward(encoded)
    for p in ps:
        if abs(p[1]-p[0]) > 0.1:
            hi = 9
        if p[1] > p[0]:
            pred.append(1)
        else:
            pred.append(0)

    score = roc_auc_score(labels.tolist(), pred)
    switch_score = roc_auc_score(switch(labels.tolist()), pred)
    print(f':clown: {score} and {switch_score} - {sum(pred)} vs {len(pred)}')
    return max(score, switch_score)


def test_if_learn(tensor, autoencoder, ae_optimizer):
    initial_params = {}
    for name, param in autoencoder.named_parameters():
        initial_params[name] = param.data.clone()

    tensor.backward()
    ae_optimizer.step()

    params_changed = any(
        not torch.equal(initial_params[name], param.data)
        for name, param in autoencoder.named_parameters()
    )

    return params_changed

def graph_reconstruction(model, normal_train_data, anomaly_train_data, suffix=""):
    temp = model.encoder(normal_train_data)
    normal_encoded = _sample(temp[0], temp[1])

    temp = model.encoder(anomaly_train_data)
    anomaly_encoded = _sample(temp[0], temp[1])

    print(pd.DataFrame(normal_encoded.cpu().detach().numpy()).describe())
    print(pd.DataFrame(anomaly_encoded.cpu().detach().numpy()).describe())

    reconstruction = model.predict(normal_train_data)[0]
    train_loss = tf.keras.losses.mae(reconstruction.cpu().detach(), normal_train_data.cpu().detach())

    reconstruction_a = model.predict(anomaly_train_data)[0]
    train_loss_a = tf.keras.losses.mae(reconstruction_a.cpu().detach(), anomaly_train_data.cpu().detach())

    train_loss = [float(l) for l in train_loss]
    plt.hist(train_loss)

    train_loss_a = [float(l) for l in train_loss_a]
    plt.hist(train_loss_a)
    plt.savefig(f'reconstruction_semivae{suffix}.png')
    plt.clf()

def ssd(x, y):
    return torch.sum((x - y) ** 2)

def sad(x, y):
    return torch.sum(torch.abs(x - y))

def gen_edge_index(graph: dgl.data.DGLDataset):
    adj = graph.adj()
    adj = sp.csr_matrix(np.array(adj.cpu().to_dense()))
    # feat = sp.csr_matrix(graph.ndata['feat'])

    # adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    # adj_norm = adj_norm.toarray()
    adj = adj + sp.eye(adj.shape[0])
    adj = adj.toarray()
    # feat = feat.toarray()
    adj_t = adj.transpose()
    adj_t = torch.LongTensor(adj_t)
    adj = torch.LongTensor(adj)
    adj_sym = torch.add(adj, adj_t)

    edge_exist = (adj_sym >1).nonzero(as_tuple=True)
    adj_sym[edge_exist] = 1

    adj_label = sp.coo_matrix(adj_sym)
    indices = np.vstack((adj_label.row, adj_label.col))
    adj_label = torch.LongTensor(indices)

    return adj_label

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()