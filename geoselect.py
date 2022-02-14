import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

#e = torch.load("attmat_ori_citeseer_3001.pt")
#s2 = torch.load("s2_citeseer.pt")
#adj_ori = torch.load("adj_citeseer_ori_300.pt")

#e = torch.load("attmat_ori_citeseer_320.pt")
#s2 = torch.load("s2_citeseer_2.pt")
#adj_ori = torch.load("adj_ori_citeseer_320.pt")


adj_ori = torch.load("adj_ori_cora_40.pt").cpu()
v = torch.load("alpha0_ori_cora_40.pt").cpu()
i = torch.load("edge_index_ori_cora_40.pt").cpu()
a = torch.sparse_coo_tensor(i, v.T[0], [adj_ori.size(dim=0), adj_ori.size(dim=0)])
e = a.to_dense()

# e = torch.load("attmat_ori_cora_300.pt")
# s2 = torch.load("s2_cora_2.pt")
# adj_ori = torch.load("adj_ori_cora_300.pt")

print(e)
print(e.size())
print(adj_ori.size())
print(torch.count_nonzero(adj_ori))


# attention_full = F.softmax(e, dim=1)
attention_full = e

attention_full_np = attention_full.detach().cpu().numpy()

adj_ori_np = adj_ori.detach().cpu().numpy()

# cora: 1% is 27, 5% is 135, 10% is 271, adj has 10556 edges
# citeseer: 1% is 33, 5% is 166, 10% is 331, adj has 9228 edges, has isolated nodes

"""
for i in range(0, 2708):
    index = int(s2[i])
    if index == 0:
        attention_full_np[i,:]=0
        continue
    partition = np.partition(attention_full_np[i].flatten(), -135)[-135]
    for id in np.ndindex(attention_full_np[i].shape):
        if attention_full_np[i][id] < partition:
            attention_full_np[i][id] = 0
"""


"""
for i in np.ndindex(adj_ori_np.shape):
    if adj_ori_np[i] != 0:
        attention_full_np[i] = 0
"""
"""
for i in range(0, 3327):
    partition = np.partition(attention_full_np[i].flatten(), -333)[-333]
    for id in np.ndindex(attention_full_np[i].shape):
        if attention_full_np[i][id] < partition:
            attention_full_np[i][id] = 0
"""
negative_inf = -math.inf
for i in np.ndindex(adj_ori_np.shape):
    if adj_ori_np[i] != 0:
        attention_full_np[i] = negative_inf


partition = np.partition(attention_full_np.flatten(), -105560)[-105560]
for id in np.ndindex(attention_full_np.shape):
    if attention_full_np[id] < partition:
        attention_full_np[id] = negative_inf


attention_full = torch.from_numpy(attention_full_np).type(torch.float32)

# attention_full = attention_full + adj_ori

one_vec = torch.ones_like(attention_full)

adj_resize_005 = torch.where(attention_full > negative_inf, one_vec, 0*one_vec)

adj_resize_005 = adj_resize_005 + adj_ori

# symmetry
# adj_resize_006 = adj_resize_005 + adj_resize_005.T.multiply(adj_resize_005.T > adj_resize_005) - adj_resize_005.multiply(adj_resize_005.T > adj_resize_005)

# print("check symmetry", torch.all(adj_resize_006.transpose(0, 1) == adj_resize_006))

s1 = torch.count_nonzero(adj_resize_005)
print(s1, "new adj non zero")

# s2 = torch.count_nonzero(adj_resize_006)
# print(s2, "symmetric adj non zero")

torch.save(adj_resize_005, "adj_resize_cora_52.pt")
print("execution ends")




