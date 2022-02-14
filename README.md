# GraphSearch

(1) Dataset: Cora
A citation network with 2708 publications in computer science. Each publication is a node and each node has 1433 features. Each feature is a representative word in the paper. The cora dataset consists of 13264 edges originally, which means around 4.89 edges per node.

In the cora.cites file, citations are included as edges in the graph. In the cora.content file, nodes, features and labels are shown.

(2) The gat network is cited from:
A. GAT in pytorch: https://github.com/Diego999/pyGAT
B: GAT in python: https://github.com/PetarV-/GAT
C: Original Publication: Graph Attention Networks (Veličković et al., ICLR 2018): https://arxiv.org/abs/1710.10903

(3) The modified GAT mainly contains several python files, which are layer01.py, models_modi.py, train.py, graph_modi.py, and utils.py, select.py.

layer01.py: 
  GATlayer: normal gat layer, used for training
  GATactilayer: colaborate with graph_modi.py, load the pretrained model and modify the adjacency matrix of the original graph.

models_modi.py:
  GAT: link to normal GATlayer, used for training
  GATacti: link to GATactilayer, collaborate with graph_modi.py, ensure the the pretrained model works fine.
 
train.py:
  cound be used for training, or load the resized adjacency matrix for training.

graph_modi.py:
  load the pretrained model, call GATacti class in models_modi.py and GATactilayer class in layer01.py. Modify the original adjancency matrix of cora graph and save the modified one as "adj_resize.pt".
 
utils.py:
  general utils such as data loader, accuracy function, and so on.
  
select.py:
resize the adjacency matrix with different number of edges.

geotrain.py:
pytorch-geometric scripts to run GAT neural networks, default settings are 30 runs enabling early stop features.

gatv2train.py:
pytorch-geometric scripts to run GATv2, default settings are 30 runs enabling early stop features.

gatv2train_load.py:
1 layer 1 head GATv2 to search for the best epoch and save the attention weights.

gatv2modi.py:
1 layer 1 head GATv2 to revise the adjacency matrix, collaborating with gatv2_modi.py.

geoselect.py:
revise the adjacency matrix according to the attention weights for GATv2.

