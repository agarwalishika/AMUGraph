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
flags.DEFINE_string('dataset', 'amazon', 'dataset name')
flags.DEFINE_bool('augment', True, 'True to enable augmentation')
FLAGS(sys.argv)

# Model
class Network_exploitation(nn.Module):
    def __init__(self, dim, hidden_size=100, k=10):
        super(Network_exploitation, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, k)
        self.cuda()

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))
    
    
class Network_exploration(nn.Module):
    def __init__(self, dim, hidden_size=100, k=10):
        super(Network_exploration, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, k)
        self.cuda()

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))

def EE_forward(net1, net2, x):

    x.requires_grad = True
    f1 = net1(x)
    net1.zero_grad()
    f1.sum().backward(retain_graph=True)
    dc = torch.cat([p.grad.flatten().detach() for p in net1.parameters()])
    dc = block_reduce(dc.cpu(), block_size=250, func=np.mean)
    dc = torch.from_numpy(dc).to(x.device)
    f2 = net2(dc)
    return f1, f2, dc

def train_NN_batch(model, X, Y, dc, num_epochs=64, lr=0.0005, batch_size=256, num_batch=4):
    model.train()
    X = torch.cat(X).float()
    Y = torch.stack(Y).float().detach()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num = X.size(1)

    for _ in range(num_batch):
        index = np.arange(len(X))
        np.random.shuffle(index)
        index = index[:batch_size]
        for _ in range(num_epochs):
            batch_loss = 0.0
            for i in index:
                x, y = X[i].to(device), Y[i].to(device)
                y = torch.reshape(y, (1,-1))
                #HEREpred = model(x, dataset, dc).view(-1)
                pred = model(x)

                optimizer.zero_grad()
                loss = torch.mean((pred - y) ** 2)
                loss.backward(retain_graph=True)
                optimizer.step()

                batch_loss += loss.item()
            
            if batch_loss / num <= 1e-3:
                return batch_loss / num

    return batch_loss / num

# Training/Testing script
ANOMALY_LABEL = 1
BENIGN_LABEL = 0
LAMBDA = 0.5
device = 'cuda'
if 'yelp' in FLAGS.dataset:
    INPUT_DIM = 139
    PARAM_FILE = 'soft_labelers/yelp_basic_mlp_params.pt'
    EXPLORE_DIM = 57
elif 'amazon' in FLAGS.dataset:
    INPUT_DIM = 133
    PARAM_FILE = 'soft_labelers/amazon_0.898883696871139basic_mlp_params.pt'
    EXPLORE_DIM = 55
elif 'cora' in FLAGS.dataset:
    INPUT_DIM = 139
    PARAM_FILE = 'soft_labelers/cora_0.61_basic_mlp_params.pt'
    EXPLORE_DIM = 279
elif 'pubmed' in FLAGS.dataset:
    INPUT_DIM = 139
    PARAM_FILE = 'soft_labelers/pubmed_0.8019_basic_mlp_params.pt'
    EXPLORE_DIM = 57

# Training/Testing script
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

def run(n=5000, margin=6, budget=0.05, num_epochs=10, dataset_name="covertype", explore_size=0, begin=0, lr=0.0001, j=0, mu=1000, gamma=1000):
    print(f'augment = {FLAGS.augment}')
    print(f'Loading in the dataset {FLAGS.dataset}...')
    train_dataset = AMUGraphDataset(train=True, dataset=FLAGS.dataset)
    test_dataset = AMUGraphDataset(train=False, dataset=FLAGS.dataset)
    print('Done loading it!')

    confidence_labeler = MLP2(input_dim=INPUT_DIM)
    confidence_labeler.load_state_dict(torch.load(PARAM_FILE))
    confidence_labeler.to(device)
    X = train_dataset.x

    # X = np.array(X)[:n]
    # if len(X.shape) == 3:
    #     N, h, w = X.shape
    #     X = np.reshape(X, (N, h*w))[:n]
    # Y = np.array(Y)
    # Y = (Y.astype(np.int64) - begin)[:n]

    test_x = test_dataset.x
    test_y = test_dataset.y
    
    test_dataset = TensorDataset(test_x, test_y.to(torch.int64))

    k = 2 #len(set(Y.tolist()))
    net1 = Network_exploitation(X.shape[1], k=k).to(device)
    net2 = Network_exploration(EXPLORE_DIM, k=k).to(device)

    X1_train, X2_train, y1, y2 = [], [], [], []
    budget = int(n * budget)
    inf_time = 0
    train_time = 0
    test_inf_time = 0
    batch_size = 5
    R = 16 * batch_size
    queried_rows = []
    counter = 0

    while counter < R:
        weights = []
        indices = []
        for i in tqdm(range(n)):
            if i in queried_rows:
                continue
            # load data point
            x = X[i].view(1, -1).to(device)

            # predict via NeurONAL
            f1, f2, dc = EE_forward(net1, net2, x)
            u = f1[0] + 1 / (i+1) * f2
            u_sort, u_ind = torch.sort(u)
            i_hat = u_sort[-1]
            i_deg = u_sort[-2]
            neuronal_pred = int(u_ind[-1].item())

            # calculate weight
            weight = abs(i_hat - i_deg).item()
            weights.append(weight)
            indices.append(i)
        

        # create the distribution and sample b points from it
        i_hat = np.argmin(weights)
        w_hat = weights[i_hat]
        distribution = []
        temp = time.time()
        for x in range(len(weights)):
            if x != i_hat:
                quotient = (mu * w_hat + gamma * (weights[x] - w_hat))
                distribution.append((w_hat / quotient))
            else:
                distribution.append(0)
        distribution[i_hat] = max(1 - sum(distribution), 0)

        total = sum(distribution)
        distribution = [w/total for w in distribution]
        inf_time = time.time() - temp

        # sample from distribution
        ind = np.random.choice(indices, size=batch_size, replace=False, p=distribution)

        round_temp_y = []
        for i in ind:
            counter += 1
            x = X[i]
            x = x.view(1, -1).to(device)

            f1, f2, dc = EE_forward(net1, net2, x)
            u = f1[0] + 1 / (i+1) * f2
            u_sort, u_ind = torch.sort(u)
            i_hat = u_sort[-1]
            i_deg = u_sort[-2]
            neuronal_pred = int(u_ind[-1].item())
            
            # add predicted rewards to the sets
            X1_train.append(x)
            X2_train.append(torch.reshape(dc, (1, len(dc))))
            r_1 = r_1 = confidence_labeler.forward(x).squeeze()
            y1.append(r_1)
            round_temp_y.append(r_1)
            y2.append((r_1 - f1)[0])

            # update unlabeled set
            queried_rows.append(i)
        
        x1t_new = []
        x2t_new = []
        y1_new = []
        y2_new = []
        for i, yi in zip(ind, round_temp_y):
            for j, yj in zip(ind, round_temp_y):
                if i == j:
                    continue
                xi, xj = X[i], X[j]
                x_new = xi * yi[0] + xj * yj[1]
                r_1_new = torch.stack((yi[0], yj[1])) 
                f1, f2, dc = EE_forward(net1, net2, x_new.detach())
                x1t_new.append(x_new.view(1,-1))
                x2t_new.append(torch.reshape(dc, (1, len(dc))))
                y1_new.append(r_1_new) 
                y2_new.append(r_1_new)
        order = torch.randperm(len(x1t_new)).to(torch.int64)[:10]
        X1_train.extend(torch.stack(x1t_new)[order])
        X2_train.extend(torch.stack(x2t_new)[order])
        y1.extend(torch.stack(y1_new)[order])
        y2.extend(torch.stack(y2_new)[order])
        print(f'\tnum points: {len(X1_train)}')

        # update the model
        temp = time.time()
        train_NN_batch(net1, X1_train, y1, dc=False, lr=lr)
        train_NN_batch(net2, X2_train, y2, dc=True, lr=lr)
        train_time = train_time + time.time() - temp

        # calculate testing regret
        labels = []
        pred = []
        for i in tqdm(range(n)):
            # load data point
            try:
                x, y = test_dataset[i]
            except:
                break
            x = x.view(1, -1).to(device)

            # predict via NeurONAL
            f1, f2, dc = EE_forward(net1, net2, x)
            u = f1[0] + 1 / (i+1) * f2
            u_sort, u_ind = torch.sort(u)
            i_hat = u_sort[-1]
            i_deg = u_sort[-2]
            neuronal_pred = int(u_ind[-1].item())

            labels.append(y.item())
            pred.append(neuronal_pred)
        
        testing_acc = roc_auc_score(labels, pred)
        switch_score = roc_auc_score(switch(labels), pred)

        print(f'testing acc after {counter} queries: {max(testing_acc, switch_score)}')
        # f = open(f"results_np/{dataset_name}/NeurONAL_pool_res.txt", 'a')
        # f.write(f'testing acc after {j} queries: {testing_acc}\n')
        # f.close()
        
    # Calculating the STD for testing acc
    for _ in range(10):
        test_ind = np.arange(len(test_dataset))
        np.random.shuffle(test_ind)
        test_ind = test_ind[:n]
        labels = []
        pred = []
        for i in tqdm(test_ind):
            # load data point
            try:
                x, y = test_dataset[i]
            except:
                break
            x = x.view(1, -1).to(device)

            # predict via NeurONAL
            temp = time.time()
            f1, f2, dc = EE_forward(net1, net2, x)
            test_inf_time = test_inf_time + time.time() - temp
            u = f1[0] + 1 / (i+1) * f2
            u_sort, u_ind = torch.sort(u)
            i_hat = u_sort[-1]
            i_deg = u_sort[-2]
            neuronal_pred = int(u_ind[-1].item())

            labels.append(y.item())
            pred.append(neuronal_pred)
        
        testing_acc = roc_auc_score(labels, pred)
        switch_score = roc_auc_score(switch(labels), pred)
        
        print(f'testing acc after {counter} queries: {max(testing_acc, switch_score)}')
        # f = open(f"results/{dataset_name}/NeurONAL_pool_res.txt", 'a')
        # f.write(f'testing acc after {j} queries: {testing_acc}\n')
        # f.close()
    return inf_time, train_time, test_inf_time


device = 'cuda'
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

run()
