import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class ClassificationLayer(nn.Module):
    def __init__(self, input_dim=10):
        super(ClassificationLayer, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 2),
            nn.Sigmoid()
        )

        self.normal_dist = torch.rand((input_dim, 10))
        self.anomaly_dist = torch.rand((input_dim, 10))

    def forward(self, encoded):
        dim = encoded.shape[-1]
        pdf_norm = 0
        pdf_anom = 0
        
        ps = []
        for enc in encoded:
            for i in range(0, dim):
                pdf_norm += norm.pdf(enc[i].item(), loc=torch.mean(self.normal_dist[i]).item(), scale=torch.std(self.normal_dist[i]).item())
                pdf_anom += norm.pdf(enc[i].item(), loc=torch.mean(self.anomaly_dist[i]).item(), scale=torch.std(self.anomaly_dist[i]).item())
            pdf_norm /= dim
            pdf_anom /= dim
            
            total = pdf_norm + pdf_anom
            ps.append([pdf_norm / total, pdf_anom / total])

        return ps

    def fit(self, normal_data, anomaly_data):
        normal_data = normal_data.squeeze(1)
        anomaly_data = anomaly_data.squeeze(1)
        self.normal_dist = torch.transpose(normal_data, 0, 1)
        self.anomaly_dist = torch.transpose(anomaly_data, 0, 1)
        
        # for i in range(0, 8):
        #     data = normal_data[:, i]
        #     data = [float(l) for l in data]
        #     plt.hist(data, label='n')

        #     data = anomaly_data[:, 0]
        #     data = [float(l) for l in data]
        #     plt.hist(data, label='a')
        #     plt.legend()
        #     plt.savefig(f'distribution_{i}.png')
        #     plt.clf()
        
        '''
        data_loader = DataLoader(TensorDataset(torch.Tensor(x), torch.Tensor(y)), batch_size=batch_size, shuffle=True)
        opt = optim.Adam(self.parameters(), lr=lr)

        for _ in tqdm(range(epochs)):
            for x_batch, y_true in data_loader:
                opt.zero_grad()
                y_pred = self.forward(x_batch)
                loss = nn.BCELoss()(y_pred, y_true)
                loss.backward(retain_graph=True)
                opt.step()
        '''
        
        return
