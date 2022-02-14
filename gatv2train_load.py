import os.path as osp

import torch
import torch.nn.functional as F

import numpy as np
import random
import glob
import os

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
# from torch_geometric.nn import GATConv 
from torch_geometric.nn import GATv2Conv
from torch_sparse import SparseTensor

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]
#adj = torch.load('/home/liuchang/exp/gat/mlp1/adj_resize_citeseer_323.pt')

# change to symmetry
# adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
# print("check symmetry", torch.all(adj.transpose(0, 1) == adj))

#adj = SparseTensor.from_dense(adj)
#row, col, value = adj.coo()
#adj = torch.stack((row, col), dim=0)
#print(data.edge_index)
#print(adj)
#for i in range(0, 100):
#    print(adj[0][i], "->")
#    print(adj[1][i])

# set the random seed, default is 72
seed = 72
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)



class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = GATv2Conv(in_channels, out_channels, heads=1, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
#        self.conv2 = GATv2Conv(8 * 8, out_channels, heads=1, concat=False,
#                             dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
#        x = F.dropout(x, p=0.6, training=self.training)
#        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
#        self.conv2.reset_parameters()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dataset.num_features, dataset.num_classes).to(device)
data = data.to(device)
# block the next sentence if adj is not loaded
#adj = adj.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)    
#    out = model(data.x, adj)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
    loss.backward()
    optimizer.step()
    
    return val_loss.data.item()

@torch.no_grad()
def test(data):
    model.eval()
    # change data.edge_index to adj
#    out, accs = model(data.x, adj), []
    out, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        acc = float((out[mask].argmax(-1) == data.y[mask]).sum() / mask.sum())
        accs.append(acc)
    return accs

test_accs = []
for run in range(1, 2):
    model.reset_parameters()

    loss_values = []
    bad_counter = 0
    best = 10001
    best_epoch = 0
    final_acc = 0
    best_acc = 0
    target_epoch = 0

    for epoch in range(1, 10001):
        loss_values.append(train(data))
        train_acc, val_acc, test_acc = test(data)
    
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1
        
        if val_acc > best_acc:
            best_acc = val_acc
            final_acc = test_acc
            
            # load the best epoch
            target_epoch = epoch
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_values,
            }, '{}.tar'.format(epoch))

            
#        print(test_acc)
#        print(f'{test_acc:.4f}')
#        print(bad_counter)
#        print("--------------")

        if bad_counter == 100:
            print(f'Run {run:02d}:' 
                f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                f'Test: {test_acc:.4f}')
            test_accs.append(final_acc)
            break
    
        if epoch > 10000:
            print(f'Run {run:02d}:'
                f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                f'Test: {test_acc:.4f}')
            test_accs.append(final_acc)

    files = glob.glob('*.tar')
    for file in files:
        epoch_enum = int(file.split('.')[0])
        if epoch_enum != target_epoch:
            os.remove(file)

test_acc = torch.tensor(test_accs)
print('============================')
print(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')
