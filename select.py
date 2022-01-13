import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

e = torch.load("attmat_ori_002.pt")
s2 = torch.load("s2.pt")

attention_full = F.softmax(e, dim=1)

attention_full_np = attention_full.detach().cpu().numpy()

# 1% is 27, 5% is 135, 10% is 271, adj has 10556 edges
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
for i in range(0, 2708):
    partition = np.partition(attention_full_np[i].flatten(), -271)[-271]
    for id in np.ndindex(attention_full_np[i].shape):
        if attention_full_np[i][id] < partition:
            attention_full_np[i][id] = 0
"""

"""
# to avoid the collision of attention edges and original edges

for i in np.ndindex(adj_ori_np.shape):
    if adj_ori_np[i] != 0:
        attention_full_np[i] = 0
"""
partition = np.partition(attention_full_np.flatten(), -62560)[-62560]
for id in np.ndindex(attention_full_np.shape):
    if attention_full_np[id] < partition:
        attention_full_np[id] = 0


attention_full = torch.from_numpy(attention_full_np).type(torch.float32)

# attention_full = attention_full + adj_ori

one_vec = torch.ones_like(attention_full)

adj_resize_005 = torch.where(attention_full > 0, one_vec, 0*one_vec)

adj_resize_005 = adj_resize_005 + adj_ori

s1 = torch.count_nonzero(adj_resize_005)
print(s1, "new adj non zero")

torch.save(adj_resize_005, "adj_resize_300511.pt")
print("execution ends")
