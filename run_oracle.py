import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from sklearn.preprocessing import MinMaxScaler
from torch import optim, nn
from tqdm import tqdm
import sys
from utils import MLP
from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_labels', 75, 'Number of queried labels per epoch')
flags.DEFINE_string('dataset', 'amazon_embeddings', 'yelp_embeddings|amazon_embeddings|cora_embeddings')
FLAGS(sys.argv)

dataset = FLAGS.dataset

df = pd.read_csv(f'{dataset}_embeddings/final_embeddings0.csv', sep=',')

for i in range(1,20):
    m = pd.read_csv(f'{dataset}_embeddings/final_embeddings{i}.csv', sep=',')
    df = pd.concat([df, m])

df = df.sample(frac=1).reset_index(drop=True)
df = df.drop(columns=['key'])
label = df.pop('label')
df.insert(0, '0', label)

x_train, x_test, y_train, y_test = train_test_split(df.values, df.values[:,0:1], test_size=0.4, random_state=111)
scaler = MinMaxScaler()
data_scaled = scaler.fit(x_train)
train_data_scaled = x_train #data_scaled.transform(x_train)
test_data_scaled = x_test #data_scaled.transform(x_test)

normal_train_data = torch.Tensor(pd.DataFrame(train_data_scaled).add_prefix('c').query('c0 == 0').values[:,1:])
normal_train_labels = torch.Tensor(normal_train_data.shape[0]*[1,0]).reshape(-1,2)

anomaly_train_data = torch.Tensor(pd.DataFrame(train_data_scaled).add_prefix('c').query('c0 > 0').values[:, 1:])
anomaly_train_labels = torch.Tensor(anomaly_train_data.shape[0]*[0,1]).reshape(-1,2)

normal_test_data = torch.Tensor(pd.DataFrame(test_data_scaled).add_prefix('c').query('c0 == 0').values[:,1:])
normal_test_labels = torch.Tensor(normal_test_data.shape[0]*[1,0]).reshape(-1,2)

anomaly_test_data = torch.Tensor(pd.DataFrame(test_data_scaled).add_prefix('c').query('c0 > 0').values[:, 1:])
anomaly_test_labels = torch.Tensor(anomaly_test_data.shape[0]*[0,1]).reshape(-1,2)

train_data = torch.cat((normal_train_data, anomaly_train_data))
train_labels = torch.cat((normal_train_labels, anomaly_train_labels))

test_data = torch.cat((normal_test_data, anomaly_test_data))
test_labels = torch.cat((normal_test_labels, anomaly_test_labels))

def train_one_model(epochs=48, lr=1e-2):
    mlp = MLP(input_dim=train_data.shape[1], num_layers=1)
    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    order = torch.randperm(train_data.shape[0])
        
    for _ in tqdm(range(epochs)):
        for i in order:
            ind = i.item()
            prob = mlp.forward(train_data[ind])
            loss = F.binary_cross_entropy(prob, train_labels[ind])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    # testing
    pred = []
    labels = []
    order = torch.arange(train_data.shape[0])
    for i in order:
        ind = i.item()
        prob = mlp.forward(train_data[ind])
        if prob[0].item() > prob[1].item():
            pred.append(0)
        else:
            pred.append(1)
        
        if train_labels[i][0].item() > train_labels[i][1].item():
            labels.append(0)
        else:
            labels.append(1)

    train_score = roc_auc_score(labels, pred)
    print('training score:',train_score)

    pred = []
    labels = []
    order = torch.arange(test_data.shape[0])
    for i in order:
        ind = i.item()
        prob = mlp.forward(test_data[ind])
        if prob[0].item() > prob[1].item():
            pred.append(0)
        else:
            pred.append(1)
        
        if test_labels[i][0].item() > test_labels[i][1].item():
            labels.append(0)
        else:
            labels.append(1)

    score = roc_auc_score(labels, pred)
    print('score:',score)
    return train_score, score, mlp

score = 0

for _ in range(5):
    _, temp, mlp = train_one_model()
    if temp > score:
        score = temp
        params_file = dataset + f'_{score}_basic_mlp_params.pt'
        torch.save(mlp.state_dict(), params_file)
with open('run_basic_mlp_results.txt', 'a+') as f:
    f.write(f'{FLAGS.dataset}|{score}\n')

#torch.save(mlp.state_dict(), params_file)
