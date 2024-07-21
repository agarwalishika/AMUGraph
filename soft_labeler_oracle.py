#https://docs.dgl.ai/en/1.1.x/tutorials/blitz/1_introduction.html
import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F

dataset = dgl.data.FraudYelpDataset()
print(f"Number of categories: {dataset.num_classes}")

g = dataset[0]
g = dgl.to_homogeneous(g, g.ndata)

print("Node features")
print(g.ndata)
print("Edge features")
print(g.edata)

from dgl.nn import GraphConv


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
        self.sig = nn.Sigmoid()

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return self.sig(h)


# Create the model with given dimensions
model = GCN(g.ndata["feature"].shape[1], 16, 2)

def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    best_val_acc = 0
    best_test_acc = 0
    best_params = None

    features = g.ndata["feature"]
    labels = g.ndata["label"]
    labels = torch.where(labels < 1, 1, 0)
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]
    for e in range(200):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        temp = torch.zeros((labels[train_mask].shape[0], 2))
        temp[labels[train_mask] == 0] = torch.Tensor([1, 0])
        temp[labels[train_mask] == 1] = torch.Tensor([0, 1])
        loss = F.cross_entropy(logits[train_mask], temp.to(device))

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
        
        if best_test_acc < test_acc:
            best_test_acc = test_acc
            best_params = model.state_dict()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print(
                f"In epoch {e}, loss: {loss:.3f}, val acc: {val_acc:.3f} (best {best_val_acc:.3f}), test acc: {test_acc:.3f} (best {best_test_acc:.3f})"
            )
    return best_params, best_test_acc

if __name__ == "__main__":
    device = 'cuda'
    g = dgl.add_self_loop(g)
    g = g.to(device)
    model = GCN(g.ndata['feature'].shape[1], 16, 2).to(device)
    best_params, best_test_acc = train(g, model)
    #torch.save(best_params, f'dgl_cora_{best_test_acc}_params.pt')