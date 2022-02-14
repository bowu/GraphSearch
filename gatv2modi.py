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
# from torch_geometric.nn import GATv2Conv
from torch_sparse import SparseTensor
from gatv2_modi import GATv2Conv

dataset = 'Cora'
path1 = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path1, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

# calculate full graph attention weights
r = data.x.size(dim=0)
k = torch.arange(0, r)
k1 = k.repeat(r, 1).transpose(1, 0).flatten()
k2 = k.repeat(1, r)
adj = torch.stack((k1, k2[0]), dim=0)
# torch.save(adj, "edge_index_ori_cora_40.pt")
# print(adj.size())
# save the original adjacency matrix
# i = data.edge_index
# v = torch.ones([i[0].size()[0]], dtype=torch.float64)
# a = torch.sparse_coo_tensor(i, v, [data.x.size(dim=0), data.x.size(dim=0)])
# d = a.to_dense()
# torch.save(d, "adj_ori_cora_40.pt")


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
adj = adj.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

checkpoint = torch.load('126.tar')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss_value = checkpoint['loss']

#model.eval()
#out = model(data.x, data.edge_index)
#loss_test = F.nll_loss(out[data.test_mask], data.y[data.test_mask])
#acc_test = accuracy(output[idx_test], labels[idx_test])
#print("Test set results:",
#        "loss= {:.4f}".format(loss_test.data.item()),
#        "accuracy= {:.4f}".format(acc_test.data.item()))

#train_acc, val_acc, test_acc = test(data)
#print(test_acc)



# def train(data):
#    model.train()
#    optimizer.zero_grad()
#    out = model(data.x, data.edge_index)    
#    out = model(data.x, adj)
#    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
#    val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
#    loss.backward()
#    optimizer.step()
    
#    return val_loss.data.item()

@torch.no_grad()
def test(data):
    model.eval()
    # change data.edge_index to adj
    out, accs = model(data.x, data.edge_index), []
#    out, accs = model(data.x, adj), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        acc = float((out[mask].argmax(-1) == data.y[mask]).sum() / mask.sum())
        accs.append(acc)
    return accs

train_acc, val_acc, test_acc = test(data)
print(test_acc)



"""
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
"""
