import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx

def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(path="./data/inddata/", dataset="pubmed"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} ind dataset...'.format(dataset))

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}ind.pubmed.{}".format(path, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}ind.pubmed.test.index".format(path))
    test_idx_range = np.sort(test_idx_reorder)

    # only for citeseer
    # Fix citeseer dataset (there are some isolated nodes in the graph)
    # Find isolated nodes, add them as zero-vecs into the right position

#    test_idx_range_full = range(min(test_idx_reorder),
#                                    max(test_idx_reorder) + 1)
#    tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
#    tx_extended[test_idx_range - min(test_idx_range), :] = tx
#    tx = tx_extended
#    ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
#    ty_extended[test_idx_range - min(test_idx_range), :] = ty
#    ty = ty_extended


    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
#    adj = torch.load('adj_resize_citeseer_003.pt')
#    print("citeseer adj size", adj.size())

#    adj = sp.coo_matrix(adj, shape=(3327, 3327), dtype=np.float32)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
  
    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
#   adj= normalize_adj(adj)
#    idx_train = torch.LongTensor(range(len(y)))
#    idx_val = torch.LongTensor(range(len(y),len(y) + 500))
#    idx_test = torch.LongTensor(test_idx_range.tolist())

#    print("test_idx_range", len(test_idx_range))
    
    print("leny", len(y))
    print("test_idx_range", len(test_idx_range))

    
    idx_train = range(200)
    idx_val = range(200, 700)
    idx_test = range(1000, 1500)



    adj = torch.FloatTensor(np.array(adj.todense()))
 
    torch.save(adj, 'adj_original_pubmed.pt')
    print('adj_pubmed get')
 
 
    features = torch.FloatTensor(features.todense())
    labels = torch.LongTensor(np.where(labels)[1])
#    print("adj", adj)
#    print("features", features)
#    print("labels", labels)



    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
