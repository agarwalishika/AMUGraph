import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import networkx as nx
import os
from tqdm import tqdm
import sqlite3
import numpy as np
import random
import dgl
from dgl.data import FraudYelpDataset

class AMUGraphDataset(Dataset):
    def __init__(self, dataset="", train=True, dataset_label=-1):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset

        print(f'Reading from {self.dataset}...')

        meta = pd.read_csv(f'{self.dataset}_embeddings/final_embeddings0.csv', sep=',')

        for i in range(1,20):
            m = pd.read_csv(f'{self.dataset}_embeddings/final_embeddings{i}.csv', sep=',')
            meta = pd.concat([meta, m])
        
        meta = meta.sample(frac=1).reset_index(drop=True)

        '''meta = pd.read_csv("ecg_final.txt", sep='  ', header=None)
        meta = meta.add_prefix('c')
        label = meta.pop('c0')
        label = [1 if x >= 3 else 0 for x in label]
        meta.insert(140, 'c0', label)'''

        train_size = int(0.75 * len(meta))
        test_size = int(0.25 * len(meta))
        self.features = meta.iloc[:, 1:-1]
        

        if train:
            features = meta.iloc[:train_size, 1:-1]
            labels = meta.iloc[:train_size, -1]
        else:
            features = meta.iloc[train_size:train_size+test_size, 1:-1]
            labels = meta.iloc[train_size:train_size+test_size, -1]

        self.node_to_str = meta.iloc[:, 0].reset_index(drop=True)

        self.str_to_node = pd.DataFrame(self.node_to_str.copy())
        self.str_to_node.columns = ['name']
        self.str_to_node = self.str_to_node.reset_index()
        self.str_to_node = self.str_to_node.set_index('name')

        self.k = 2

        self.x = torch.tensor(features.values, dtype=torch.float32).to(device)
        self.y = torch.tensor(labels.values).to(device)

        if dataset_label >= 0:
            ind = torch.nonzero(self.y==dataset_label).reshape(-1)
            self.y = self.y[ind]
            self.x = self.x[ind]


        # if not os.path.exists(f'{self.dataset}neighbors.txt'):
        #     self.print_neighbors()

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        index = int(index)
        return self.x[index], self.y[index], index
    
    def get_neighbors(self, node):
        conn = sqlite3.connect(f'{self.dataset}neighbors.db')
        cursor = conn.cursor()
        cursor.execute(f'SELECT neighbors FROM neighbors WHERE node_num = {node}')
        result = cursor.fetchone()
        conn.close()

        neighbors = result[0]
        neighbors = neighbors.split(',')
        neighbors = [int(i) for i in neighbors if i != '']
        neighbors = [i for i in neighbors if i < self.features.shape[0]]

        # for debugging purposes
        with open('progress.txt', 'a') as f:
            f.write(f'{node}, ')

        return neighbors
    
    def get_random_fast(self, neighbors):
        do_not_pick = {}
        for n in neighbors:
            do_not_pick[n] = 1
        
        random_nodes = []
        while len(random_nodes) != len(neighbors):
            x = random.randint(0,self.features.shape[0] - 1)
            try:
                do_not_pick[x]
            except KeyError:
                random_nodes.append(x)
            
        return random_nodes

    def get_random(self, node, num_nodes):
        conn = sqlite3.connect(f'{self.dataset}random.db')
        cursor = conn.cursor()
        cursor.execute(f'SELECT neighbors FROM random_neighbors WHERE node_num = {node}')

        result = cursor.fetchone()
        conn.close()

        nodes = result[0]
        nodes = nodes.split(',')
        nodes = [int(i) for i in nodes if i != '']

        return np.random.choice(nodes, num_nodes)
        '''while len(samples) < num_nodes:
            samp = self.node_to_str.sample().index[0]
            if samp not in samples or samp not in neighbors:
                samples.append(samp)'''
        
        return samples
    
    def print_neighbors(self):
        print('Loading all the neighbors')
        '''f = open(f'{self.dataset}graph.txt', 'rb')
        G = pickle.load(f)
        f.close()'''

        dataset = FraudYelpDataset()
        g = dataset[0] 
        G = dgl.to_homogeneous(g)
        G = dgl.to_networkx(G)

        nodes = G.nodes
        for n in tqdm(nodes):
            neighbors = nx.single_source_shortest_path_length(G, n, self.k)
            neighbors = [k for k,v in neighbors.items() if n is not k] #[self.str_to_node.loc[i][0] for i in neighbors if i is not n]
            neighbors = ','.join(str(n) for n in neighbors)
            f = open(f'{self.dataset}neighbors.txt', 'a')
            f.write(f'{n}: {neighbors}\n')
            f.close()
        print('Done loading all the neighbors into neighbors.txt')
        return neighbors
