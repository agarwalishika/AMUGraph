import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import random
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import sys
from absl import flags
from graph_dataset import AMUGraphDataset
from classification_layer import ClassificationLayer
import matplotlib.pyplot as plt
from semi_vae import SemiVAE
from semi_vae import _sample
from utils import *

FLAGS = flags.FLAGS
flags.DEFINE_float('medium_lower_threshold', 0.4, 'Medium Lower Threshold')
flags.DEFINE_float('medium_upper_threshold', 0.8, 'Medium Upper Threshold')
flags.DEFINE_float('budget_percentage', 0.035, 'Number of queried labels per epoch')
flags.DEFINE_integer('epochs', 16, 'Epochs')
flags.DEFINE_string('dataset', 'pubmed_embeddings', 'dataset name')
flags.DEFINE_integer('latent_dim', 8, 'latent dimension')
FLAGS(sys.argv)

torch.manual_seed(123)    # reproducible

# Hyper Parameters
EPOCH = FLAGS.epochs
BATCH_SIZE = 1
LR = 0.01         # learning rate
EMBED_DIM = 10
HIDDEN_DIM=5
SIZE=30
MEDIUM_LOWER_THRESHOLD = FLAGS.medium_lower_threshold
MEDIUM_UPPER_THRESHOLD = FLAGS.medium_upper_threshold
S_A_IND = 0
S_B_IND = 1
ANOMALY_LABEL = 1
BENIGN_LABEL = 0
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if 'yelp' in FLAGS.dataset:
    INPUT_DIM = 139
    PARAM_FILE = 'soft_labelers/yelp_basic_mlp_params.pt'
elif 'amazon' in FLAGS.dataset:
    INPUT_DIM = 133
    PARAM_FILE = 'soft_labelers/amazon_0.8954748564122422basic_mlp_params.pt'
elif 'cora' in FLAGS.dataset:
    INPUT_DIM = 139
    PARAM_FILE = 'soft_labelers/cora_basic_mlp_params.pt'
elif 'pubmed' in FLAGS.dataset:
    INPUT_DIM = 139
    PARAM_FILE = 'soft_labelers/pubmed_0.8019_basic_mlp_params.pt'

print('Loading in the dataset...')
train_dataset = AMUGraphDataset(train=True, dataset=FLAGS.dataset)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
test_dataset = AMUGraphDataset(train=False, dataset=FLAGS.dataset)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)
print('Done loading it!')

autoencoder = SemiVAE([INPUT_DIM], FLAGS.latent_dim, [64, 32, 16])
autoencoder.to(device)
for param in autoencoder.parameters():
    param.requires_grad_(True)

classification_layer = ClassificationLayer(input_dim=HIDDEN_DIM)
classification_layer.to(device)

rec_loss = ssd

confidence_labeler = MLP(input_dim=INPUT_DIM)
confidence_labeler.load_state_dict(torch.load(PARAM_FILE))
confidence_labeler.to(device)


QUERY_BUDGET = int(FLAGS.budget_percentage * len(train_dataset) / FLAGS.epochs)

### Autoencoder ###
ae_params = 'debug_autoencoder_params.pt'
cl_params = 'debug_classification_layer_params.pt'
validation_scores = []

def train_class_layer():
    l_hat = []
    cl_soft_labels = []
    print("\tCollecting reconstruction losses for all the training points")
    # Run through to get the reconstruction losses
    normal_encoded = []
    anomaly_encoded = []
    for feat, _, _ in tqdm(train_loader):
        feat.requires_grad = True
        dec, mu, var = autoencoder(feat)
        l_hat.append(rec_loss(dec, feat).item())
        
    l_min = min(l_hat)
    l_max = max(l_hat)
    l_hat = [(l - l_min) / (l_max - l_min) for l in l_hat]

    for feat, _, ind in tqdm(train_loader):
        feat.requires_grad = True
        _, mu, var = autoencoder(feat)
        enc = _sample(mu, var)
        if l_hat[ind] < 0.5:
            normal_encoded.append(enc)
        else:
            anomaly_encoded.append(enc)
    
    normal_encoded = torch.stack(normal_encoded).to(device)
    anomaly_encoded = torch.stack(anomaly_encoded).to(device)
    
    classification_layer.fit(normal_encoded, anomaly_encoded)

if not os.path.exists(ae_params):

    print("### Training autoencoder ###")
    for epoch in range(EPOCH):
        print(f"Epoch {epoch+1}/{EPOCH}")
        l_hat = []
        print("\tCollecting reconstruction losses for all the training points")
        # Run through to get the reconstruction losses
        for feat, _, _ in tqdm(train_loader):
            feat.requires_grad = True
            dec, mu, var = autoencoder(feat)
            l_hat.append(rec_loss(dec, feat))
        
        l_hat, index = normalize(l_hat)
            
        ### Active Learning ###
        print("\tQuerying the difficult points")
        queries = torch.where((l_hat >= MEDIUM_LOWER_THRESHOLD) & (l_hat <= MEDIUM_UPPER_THRESHOLD))[0]
        queries = queries[torch.randperm(len(queries))[:QUERY_BUDGET]]

        labeled = []
        for q in tqdm(queries):
            # querying
            point = train_dataset.x[index[q]]
            prob = confidence_labeler.forward(point)
            labeled.append(torch.Tensor([index[q], prob[0].item(), prob[1].item()]))
    
        ### Contrastive Learning ###

        # Getting positive/negative neighborhood reconstruction losses

        print("\tAugmenting data points")
        augmented_points = []
        augmented_labels = []

        # fix actual data points
        labeled = torch.stack(labeled).to(device)
        benign = torch.where(labeled[:, 1] > labeled[:, 2])[0]
        anomalies = torch.where(labeled[:, 1] < labeled[:, 2])[0]
        idx = labeled[:, 0]
        data_points = train_dataset.x[idx.int()].to(device)
        data_labels = labeled[:, 1:3]

        for b1 in benign:
            for b2 in benign:
                b1point = data_points[b1]
                b2point = data_points[b2]

                augmented_points.append(labeled[b1][1] * b1point + labeled[b2][2] * b2point)
                augmented_labels.append(torch.stack([labeled[b1][1], labeled[b2][2]]))

        for a1 in anomalies:
            for a2 in anomalies:
                a1point = data_points[a1]
                a2point = data_points[a2]

                augmented_points.append(labeled[a1][1] * a1point + labeled[a2][2] * a2point)
                augmented_labels.append(torch.stack([labeled[a1][1], labeled[a2][2]]))

        for b in benign:
            for a in anomalies:
                bpoint = data_points[b]
                apoint = data_points[a]

                augmented_points.append(labeled[b][1] * bpoint + labeled[a][2] * apoint)
                augmented_labels.append(torch.stack([labeled[b][1], labeled[a][2]]))

                augmented_points.append(labeled[b][2] * bpoint + labeled[a][1] * apoint)
                augmented_labels.append(torch.stack([labeled[a][1], labeled[b][2]]))
            
        # Get ready to backpropagate
        autoencoder.train()

        if len(augmented_points):
            augmented_points = torch.stack(augmented_points)
            augmented_labels = torch.stack(augmented_labels)

            x = torch.cat((augmented_points, data_points))
            y = torch.cat((augmented_labels, data_labels))
        else:
            x = data_points
            y = data_labels

        #idx = torch.where(y == 0)[0]
        autoencoder.fit(x, y)

        autoencoder.eval()
        train_class_layer()
        validation_scores.append(testing(test_dataset, autoencoder, classification_layer))
        classification_layer = ClassificationLayer(input_dim=HIDDEN_DIM)
        #train_normal = train_dataset.x[train_dataset.y == BENIGN_LABEL]
        #train_anomaly = train_dataset.x[train_dataset.y == ANOMALY_LABEL]
        #graph_reconstruction(autoencoder, train_normal, train_anomaly, suffix=str(epoch))

    #torch.save(autoencoder.state_dict(), ae_params)
else:
    print("Loaded old autoencoder parameters!")
    autoencoder.load_state_dict(torch.load(ae_params))

### Classification Layer ###
torch.autograd.set_detect_anomaly(True)
if not os.path.exists(cl_params):
    print('\n### Training the classification layer ###')
    train_class_layer()

    #torch.save(classification_layer.state_dict(), cl_params)

else:
    print("Loaded old classification layer parameters!")
    classification_layer.load_state_dict(torch.load(cl_params))



### Testing the architecture ###
score = testing(test_dataset, autoencoder, classification_layer)

print(f'Validation:', validation_scores)
print(f'ROC_AUC: {max(max(validation_scores), score)}')

with open('run_soft_results.txt', 'a+') as f:
    f.write(f'claecl|{FLAGS.dataset}|{FLAGS.latent_dim}|{max(max(validation_scores), score)}\n')

