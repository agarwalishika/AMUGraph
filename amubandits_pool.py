import os
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
from amu_graph import AMUGraph
from semi_vae import _sample
from semi_vae import _loss_semi_vae

FLAGS = flags.FLAGS
flags.DEFINE_float('medium_lower_threshold', 0.4, 'Medium Lower Threshold')
flags.DEFINE_float('medium_upper_threshold', 0.8, 'Medium Upper Threshold')
flags.DEFINE_float('budget_percentage', 0.035, 'Number of queried labels per epoch')
flags.DEFINE_integer('epochs', 16, 'Epochs')
flags.DEFINE_string('dataset', 'pubmed', 'dataset name')
flags.DEFINE_integer('latent_dim', 8, 'latent dimension')
flags.DEFINE_bool('augment', True, 'True to enable augmentation')
flags.DEFINE_integer('mu', 1000, 'mu=1k')
flags.DEFINE_integer('gamma', 1000, 'gamma=1k')
flags.DEFINE_float('lr', 1e-3, 'lr = 1e-3')
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
    # X = torch.cat(X).float()
    # Y = torch.stack(Y).float().detach()
    Y = Y.detach()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

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

    return

# Training/Testing script
def normalize(x):
    x_tens = torch.stack(x)
    x_sort = torch.sort(x_tens)
    val = x_sort.values
    return (val-val[0]) / (val[-1]-val[0]), x_sort.indices

def calculate_loss(f1, x, f2):
    f1_rec = ssd(f1, x)
    f2_loss = 1 / torch.abs(f2[0] - f2[1])
    u = 0.5 * f1_rec + (1 - 0.5) * f2_loss
    return u

def run(n=10000, lr=0.0001):
    mu = FLAGS.mu
    gamma = FLAGS.gamma
    print(f'HI HI HI mu = {mu}, gamma = {gamma}')

    # load the data
    amu_graph = AMUGraph(FLAGS)

    X = amu_graph.train_data[:7000]
    confidence_labeler = amu_graph.confidence_labeler

    test_x = amu_graph.test_dataset.x
    test_y = amu_graph.test_dataset.y
    test_dataset = TensorDataset(test_x, test_y.to(torch.int64))

    amubandits_scores = []

    # helper functions
    def get_exploit_params(data):
        amu_graph.autoencoder.zero_grad()
        x_hat, mu, sig = amu_graph.autoencoder.forward(data)
        z = _sample(mu, sig)
        prob = torch.Tensor(amu_graph.classification_layer.forward(z)[0]).to(device)
        loss = sum(_loss_semi_vae(data, x_hat, prob.view(1,-1), mu, sig))
        loss.backward(retain_graph=True)
        dc = torch.cat([p.grad.flatten().detach() for p in amu_graph.autoencoder.parameters()])
        dc = block_reduce(dc.cpu(), block_size=250, func=np.mean)
        dc = torch.from_numpy(dc).to(device)
        return dc 
    
    def predict(data, info=0):
        latent = amu_graph.autoencoder.forward(data)
        z = _sample(latent[1], latent[2])
        f1 = torch.Tensor(amu_graph.classification_layer.forward(z)[0]).to(device)

        dc = get_exploit_params(data)
        f2 = net2(dc)
        
        prob = f1 + f2
        pred = torch.argmax(prob).item()
        i_hat = max(prob[0], prob[1])
        i_deg = min(prob[0], prob[1])
        if info == 2:
            return i_hat, i_deg, prob, dc, f2, pred
        elif info == 1:
            return i_hat, i_deg
        else:
            return pred

    # set up networks
    k = 2
    dc = get_exploit_params(X[0].view(1,-1))
    net2 = Network_exploration(dc.shape[0], k=k).to(device)

    # set up training loop
    X1_train, X2_train, y1, y2 = [], [], [], []
    batch_size = amu_graph.QUERY_BUDGET
    R = 16
    queried_rows = []
    counter = 0
    max_testing_score = 0

    while counter < R:
        weights = []
        indices = []
        for i, x in enumerate(tqdm(X)):
            if i in queried_rows:
                continue
            x = x.view(1, -1).to(device)

            # predict via NeurONAL
            i_hat, i_deg = predict(x, info=1)

            # calculate weight
            weight = abs(i_hat - i_deg).item()
            weights.append(weight)
            indices.append(i)

        # create the distribution and sample b points from it
        i_hat = np.argmin(weights)
        w_hat = weights[i_hat]
        if w_hat == 0:
            w_hat == 1e-10
        distribution = []
        for x in range(len(weights)):
            if x != i_hat:
                quotient = (mu * w_hat + gamma * (weights[x] - w_hat))
                if quotient == 0:
                    quotient = 1e-10
                distribution.append((w_hat / quotient))
            else:
                distribution.append(0)
        distribution[i_hat] = max(1 - sum(distribution), 0)

        total = sum(distribution)
        distribution = [w/total for w in distribution]

        # sample from distribution
        try:
            ind = np.random.choice(indices, size=batch_size, replace=False, p=distribution)
        except:
            print(f'len(indices) = {len(indices)}')
            print(f'distribution = {distribution}')
            print(f'batch_size = {batch_size}')
            0/0
        round_temp_y = []
        for i in ind:
            x = X[i]
            x = x.view(1, -1).to(device)
            _, _, prob, dc, _, _ = predict(x, info=2)

            # add predicted rewards to the sets
            X1_train.append(x)
            X2_train.append(torch.reshape(dc, (1, len(dc))))
            r_1 = r_1 = confidence_labeler.forward(x).squeeze()
            y1.append(r_1)
            round_temp_y.append(r_1)
            y2.append((r_1 - prob))

            # update unlabeled set
            queried_rows.append(i)
        counter += 1
        
        # mixup
        x1t_new = []
        x2t_new = []
        y1_new = []
        y2_new = []
        for i, yi in zip(ind, round_temp_y):
            for j, yj in zip(ind, round_temp_y):
                if i == j:
                    continue
                # add to exploitation network training buffer
                xi, xj = X[i], X[j]
                x_new = (xi * yi[0] + xj * yj[1]).view(1,-1)
                r_1_new = torch.stack((yi[0], yj[1])) 
                x1t_new.append(x_new)
                y1_new.append(r_1_new) 
                
                # add to exploration network's training buffer
                _, _, prob, dc, _, _ = predict(x_new, info=2)
                x2t_new.append(torch.reshape(dc, (1, len(dc))))
                y2_new.append(r_1_new - prob)

        # construct training set
        order = torch.randperm(len(X1_train)).to(torch.int64)[:200]
        x1_training_set = torch.cat((torch.stack(X1_train).to(device)[order], torch.stack(x1t_new).to(device)), 0)
        x2_training_set = torch.cat((torch.stack(X2_train).to(device)[order], torch.stack(x2t_new).to(device)), 0)
        y1_training_set = torch.cat((torch.stack(y1).to(device)[order], torch.stack(y1_new).to(device)), 0)
        y2_training_set = torch.cat((torch.stack(y2).to(device)[order], torch.stack(y2_new).to(device)), 0)


        # update the model
        amu_graph.autoencoder.train()
        amu_graph.autoencoder.fit(x1_training_set, y1_training_set)
        temp_cl = ClassificationLayer()
        amu_graph.train_class_layer(amu_graph.train_data, temp_cl)

        # load model checkpoints
        #amu_graph.autoencoder.load_state_dict(torch.load(f'amu_graph_checkpoints/{counter-1}_ae.pt'))
        #with open(f'amu_graph_checkpoints/{counter-1}_cl.pkl', 'rb') as f:
        #    temp_cl = pickle.load(f)

        train_NN_batch(net2, x2_training_set, y2_training_set, dc=True, lr=lr)

        # calculate testing regret
        testing_acc = testing(amu_graph.test_dataset, amu_graph.autoencoder, temp_cl) #calculate_testing_acc()
        max_testing_score = max(testing_acc, max_testing_score)
        amubandits_scores.append(testing_acc)

        print(f'testing acc after {counter} queries: {testing_acc}')

        # append to real training buffer
        X1_train.extend(torch.stack(x1t_new))
        X2_train.extend(torch.stack(x2t_new))
        y1.extend(torch.stack(y1_new))
        y2.extend(torch.stack(y2_new))
        
    # Calculating the STD for testing acc
    amu_graph.train_class_layer(amu_graph.train_data, amu_graph.classification_layer)

    for _ in range(5):
        test_ind = np.arange(len(test_dataset))
        np.random.shuffle(test_ind)
        test_ind = test_ind[:n]
        testing_acc = testing(amu_graph.test_dataset, amu_graph.autoencoder, amu_graph.classification_layer) #calculate_testing_acc()
        max_testing_score = max(testing_acc, max_testing_score)
        
        print(f'testing acc after {counter*batch_size} queries: {testing_acc}')
        # f = open(f"results/{dataset_name}/NeurONAL_pool_res.txt", 'a')
        # f.write(f'testing acc after {j} queries: {testing_acc}\n')
        # f.close()
    
    with open(f'amub_results_{amu_graph.args.dataset}.txt', 'a+') as f:
        f.write(f'{counter},{amu_graph.args.latent_dim},{amu_graph.args.epochs},{max(max(amubandits_scores), max_testing_score)}\n')



device = 'cuda'
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

run()
