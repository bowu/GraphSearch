import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random

adj_ori = torch.load("adj_ori_300.pt")

adj_ran = adj_ori

idx = 0
while(idx < 100):
  a = random.randint(0, 2707)
  b = random.randint(0, 2707)
  print("a, b", a, b)
  if (a == b) or (adj_ran[a][b] != 0): continue
  adj_ran[a][b] = 1
  idx+=1

print("idx", idx)

# torch.save(adj_ran, "adj_ran_30010.pt")
s1 = torch.count_nonzero(adj_ran)
print(s1)
print("execution ends")



