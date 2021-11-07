# GraphSearch

(1) Dataset: Cora
A citation network with 2708 publications in computer science. Each publication is a node and each node has 1433 features. Each feature is a representative word in the paper.

In the cora.cites file, citations are included as edges in the graph. In the cora.content file, nodes, features and labels are shown.

(2) The gat network is cited from:
A. GAT in pytorch: https://github.com/Diego999/pyGAT
B: GAT in python: https://github.com/PetarV-/GAT
C: Original Publication: Graph Attention Networks (Veličković et al., ICLR 2018): https://arxiv.org/abs/1710.10903

(3) The modified GAT mainly contains five python files, which are layer01.py, models_modi.py, train.py, graph_modi.py, and utils.py.

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
  
(4)Results:
  first round of training: GAT reaches the best performance at the 766th epoch. /test results: loss value = 0.6526, accuracy: 0.8420.
  
  load the model in this epoch, revise the adjancency matrix, using the new adjancency matrix and original features for a second round of training.
  GAT reaches the best performance at the 724th epoch. /test results: loss_value = 0.7990, accuracy: 0.8110
  
  % the 2-hop attention matrix is simply defined as: attention2step = torch.mm(attention, attention.T), attention is the attention matrix for 1-hop.
  
  % combined attention matrix is defined as attention_combine = attention2step + attention
  
  % there are 13264 non-zero elements in attention, thus largest 13264 non-zero elemnets in attention_combine are selected.
        code:
        attention_combinenp = attention_combine.detach().cpu().numpy()
        partition = np.partition(attention_combinenp.flatten(), -13264)[-13264]
        for id in np.ndindex(attention_combinenp.shape):
            if attention_combinenp[id] < partition:
                attention_combinenp[id] = 0
        attention_combine = torch.from_numpy(attention_combinenp).type(torch.float32)
  
  % all non-zero elements in attention_combine are tranfered as new edges in resized adjacency matrix

  tried to block all diagonal element in attention2step as 0, then calculated the summation with attention.
  GAT reaches the best performance at the 723th epoch. /test results: loss_value = 0.8010, accuracy: 0.8020



  
