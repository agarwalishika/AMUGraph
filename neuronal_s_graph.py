import os
import time
import random
import numpy as np
from numpy.random import choice
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pickle
import sys
from skimage.measure import block_reduce
from absl import flags
from graph_dataset import AMUGraphDataset
from classification_layer import ClassificationLayer
from semi_vae import SemiVAE
import torch.utils.data as Data
from utils import *
from sklearn.metrics import roc_auc_score

FLAGS = flags.FLAGS
flags.DEFINE_float('medium_lower_threshold', 0.4, 'Medium Lower Threshold')
flags.DEFINE_float('medium_upper_threshold', 0.8, 'Medium Upper Threshold')
flags.DEFINE_float('budget_percentage', 0.035, 'Number of queried labels per epoch')
flags.DEFINE_integer('epochs', 16, 'Epochs')
flags.DEFINE_string('dataset', 'pubmed', 'dataset name')
flags.DEFINE_bool('augment', True, 'True to enable augmentation')
FLAGS(sys.argv)

# Model
class Network_exploitation(nn.Module):
    def __init__(self, dim, hidden_size=100, k=10):
        super(Network_exploitation, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, k)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))
    
    
class Network_exploration(nn.Module):
    def __init__(self, dim, hidden_size=100, k=10):
        super(Network_exploration, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, k)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))

def EE_forward(net1, net2, x):

    x.requires_grad = True
    f1 = net1(x)
    net1.zero_grad()
    f1.sum().backward(retain_graph=True)
    dc = torch.cat([p.grad.flatten().detach() for p in net1.parameters()])
    dc = block_reduce(dc.cpu(), block_size=51, func=np.mean)
    dc = torch.from_numpy(dc).to(x.device)
    f2 = net2(dc)
    return f1, f2, dc

def train_NN_batch(model, X, Y, num_epochs=10, lr=0.0001, batch_size=64):
    model.train()
    X = torch.cat(X).float()
    Y = torch.stack(Y).float().detach()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num = X.size(1)

    for i in range(num_epochs):
        batch_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y = torch.reshape(y, (1,-1))
            pred = model(x).view(-1)

            optimizer.zero_grad()
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
        
        if batch_loss / num <= 1e-3:
            return batch_loss / num

    return batch_loss / num

# Traning/Testing script
ANOMALY_LABEL = 1
BENIGN_LABEL = 0
LAMBDA = 0.5
device = 'cuda'
mlp_instance = MLP2
if 'yelp' in FLAGS.dataset:
    INPUT_DIM = 139
    PARAM_FILE = 'soft_labelers/yelp_basic_mlp_params.pt'
    EXPLORE_DIM = 279
elif 'amazon' in FLAGS.dataset:
    INPUT_DIM = 133
    PARAM_FILE = 'soft_labelers/amazon_0.8863368891406275_basic_mlp_params.pt'
    EXPLORE_DIM = 267
    mlp_instance = MLP
elif 'cora' in FLAGS.dataset:
    INPUT_DIM = 139
    PARAM_FILE = 'soft_labelers/cora_0.61_basic_mlp_params.pt'
    EXPLORE_DIM = 279
elif 'pubmed' in FLAGS.dataset:
    INPUT_DIM = 139
    PARAM_FILE = 'soft_labelers/pubmed_0.8019_basic_mlp_params.pt'
    EXPLORE_DIM = 279

def normalize(x):
    x_tens = torch.stack(x)
    x_sort = torch.sort(x_tens)
    val = x_sort.values
    return (val-val[0]) / (val[-1]-val[0]), x_sort.indices

def calculate_loss(f1, x, f2):
    f1_rec = ssd(f1, x)
    f2_loss = 1 / torch.abs(f2[0] - f2[1])
    u = LAMBDA * f1_rec + (1 - LAMBDA) * f2_loss
    return u

def run(margin=7, budget=0.05, num_epochs=10, dataset_name="covertype", explore_size=0, begin=0, lr=0.0001):
    print(f'augment = {FLAGS.augment}')
    print(f'Loading in the dataset {FLAGS.dataset}...')
    train_dataset = AMUGraphDataset(train=True, dataset=FLAGS.dataset)
    test_dataset = AMUGraphDataset(train=False, dataset=FLAGS.dataset)
    print('Done loading it!')

    confidence_labeler = mlp_instance(input_dim=INPUT_DIM)
    confidence_labeler.load_state_dict(torch.load(PARAM_FILE))
    confidence_labeler.to(device)
    X = train_dataset.x
    Y = train_dataset.y
    n = X.shape[0]

    test_x = test_dataset.x
    test_y = test_dataset.y
    
    test_dataset = TensorDataset(test_x, test_y.to(torch.int64))

    k = 2
    net1 = Network_exploitation(X.shape[1], k=k).to(device)
    net2 = Network_exploration(EXPLORE_DIM, k=k).to(device)

    X1_train, X2_train, y1, y2 = [], [], [], []
    budget = int(n * budget)
    batch_size = 5
    R = 16 * batch_size
    current_regret = 0
    query_num = 0
    counter = 0
    regret = []

    points = np.arange(0, n)

    i = 0
    while counter < n and query_num < budget:
        index = random.choice(np.arange(points.size))
        counter += 1
        try:
            x, y = X[points[index]], Y[points[index]]
        except:
            break
        x = x.view(1, -1).to(device)

        f1, f2, dc = EE_forward(net1, net2, x)
        u = f1[0] + 1 / (i+1) * f2
        u_sort, u_ind = torch.sort(u)
        i_hat = u_sort[-1]
        i_deg = u_sort[-2]

        ind = 0
        if abs(i_hat - i_deg) < margin * 0.1:
            i += 1
            ind = 1
            points = np.delete(points, index)

            #construct training set        
            pred = int(u_ind[-1].item())

            lbl = y.item()
            if pred != lbl:
                current_regret += 1
                reward = 0 
            else:
                reward = 1

            if ind and (query_num < budget): 
                query_num += 1

                #add predicted rewards to the sets
                X1_train.append(x)
                X2_train.append(torch.reshape(dc, (1, len(dc))))
                r_1 = torch.zeros(k).to(device)
                r_1[lbl] = 1
                y1.append(r_1) 
                y2.append((r_1 - f1)[0])

                train_NN_batch(net1, X1_train, y1, num_epochs=num_epochs, lr=lr)
                train_NN_batch(net2, X2_train, y2, num_epochs=num_epochs, lr=lr)
            
            regret.append(current_regret)
        print(f'{counter},{query_num},{budget},{num_epochs},{current_regret}')
        with open(f'results/amubs_{FLAGS.dataset}.txt', 'a+') as f:
            f.write(f'{counter},{query_num},{budget},{num_epochs},{current_regret}\n')
        
    print('-------TESTING-------')
    lim = test_x.shape[0]
    for _ in range(5):
        acc = 0
        labels = []
        preds = []
        for i in range(lim):
            x, y = test_x[i], test_y[i]
            x = x.view(1, -1).to(device)

            f1, f2, dc = EE_forward(net1, net2, x)
            u = f1[0] + 1 / (i+1) * f2
            u_sort, u_ind = torch.sort(u)
            i_hat = u_sort[-1]
            i_deg = u_sort[-2]
                
            pred = int(u_ind[-1].item())
            labels.append(y.item())
            preds.append(pred)

        score = max(roc_auc_score(labels, preds), roc_auc_score(switch(labels), preds))
        print(f'Testing accuracy: {score}\n')
        with open(f'results/amubs_{FLAGS.dataset}.txt', 'a+') as f:
            f.write(f'Testing accuracy: {score}\n')


device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
run()