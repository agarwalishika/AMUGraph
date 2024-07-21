import numpy as np
import pandas as pd
import os
from node2vec import Node2Vec
import networkx as nx
import pickle
import dgl
from dgl.data import FraudAmazonDataset, FraudYelpDataset

def get_graph():
    dataset = FraudYelpDataset()
    g = dataset[0] 
    G = dgl.to_homogeneous(g)
    G = dgl.to_networkx(G)

    return g, G

dir_name = ""

if not os.path.isfile(f'embeddings/embeddings'):
    _, G = get_graph()

    embeds = Node2Vec(G, dimensions=108, walk_length=10, num_walks=20, workers=1)
    model = embeds.fit(window=10, min_count=1, batch_words=4)
    model.wv.save_word2vec_format('embeddings')

g, G = get_graph()

embeds = pd.read_csv(f'embeddings/embeddings', sep=' ', skiprows=[0], header=None)
embeds = embeds.rename(columns={0: "key"})

for i in range(1, len(embeds.columns)):
    xmin = embeds[i].min()
    xmax = embeds[i].max()
    embeds[i] = (embeds[i] - xmin) / (xmax - xmin)

rev_embeds = embeds

graph_features = g.ndata['feature'].transpose(0,1).tolist()
for i in range(109,140):
    rev_embeds[str(i)] = graph_features[i-109]

graph_labels = g.ndata['label'].tolist()
rev_embeds['label'] = graph_labels


rev_embeds = rev_embeds.reset_index(drop=True)

# split the dataset and save it to csv files
r = int(rev_embeds.shape[0] / 20)

for i in range(0, 20):
    rev_embeds.iloc[i*r:(i+1)*r].to_csv(f'final_embeddings{i}.csv', index=False)
