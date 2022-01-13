import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

class GATlayer(nn.Module):
    #define the graph attention layer, single layer

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATlayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size = (in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain = 1.414)

        self.a = nn.Parameter(torch.empty(size = (2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain = 1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)

#        adj_diag = adj
#        torch.save(adj_diag, "adj_ori.pt")

#        adj_diag.fill_diagonal_(0)

#        s2 = torch.count_nonzero(adj_diag, dim=1)

#        torch.save(s2, "s2.pt")

        e = self._prepare_attentional_mechanism_input(Wh)

#        torch.save(e, "attmat_ori_002.pt")
#        print(e)

        zero_vec = -9e15*torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim = 1)

        attention = F.dropout(attention, self.dropout, training = self.training)
        
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        
    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T

        return self.leakyrelu(e)
    
    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.in_features) + '->' + str(self.out_features) + ')'


class GATactilayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATactilayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)


        # diagonal to zero
        adj_diag = adj
        torch.save(adj_diag, "adj_ori.pt")

        adj_diag.fill_diagonal_(0)

        s2 = torch.count_nonzero(adj_diag, dim=1)

        torch.save(s2, "s2.pt")


#       adj_original = torch.load('adj_original_cora.pt')
#        s1 = torch.count_nonzero(adj_original)
#        print("citeseer adj size", adj.size())
#        s2 = torch.count_nonzero(adj, dim=1)
#        s12 = adj[0:140, 0:140]
#        s13 = torch.count_nonzero(s12)
#        print("adj first 140", s13)



#        s2 = torch.count_nonzero(adj_original, dim=1)
#        s3 = torch.count_nonzero(attention)
#        s4 = torch.count_nonzero(attention, dim=1)
#        print('adj count', s1)
        print('adj count_at_array', s2)
        print('s2.0.1', sum(s2))
#        print('attention count_at_array', s4)


#        print("attention tensor saved")
#        torch.save(attention, 'attention.pt')

        attention2step = torch.mm(attention, attention.T)
#        attention2step = attention2step.fill_diagonal_(0)

#        print("attention2step tensor saved")
#        torch.save(attention2step, 'attention2step.pt')
       
        attention_combine = attention + attention2step
        attention_combine.fill_diagonal_(0)

#        attention_combine = attention * 0.15+ attention2step * 0.85
#        attention_combine = attention_combine.fill_diagonal_(0)

#        print("attention_combine tensor saved")
#        torch.save(attention_combine, 'attention_combine.pt')


        attention_combinenp = attention_combine.detach().cpu().numpy()
 
#        print('attention_combine_np_size', np.size(attention_combinenp))
#        print('attention_combine_np_size_0', np.size(attention_combinenp[0]))

        for i in range(0, 3327):
            index = -int(s2[i])
            if index == 0:
                attention_combinenp[i,:] = 0
                continue
            partition = np.partition(attention_combinenp[i].flatten(), index)[index]
            for id in np.ndindex(attention_combinenp[i].shape):
                if attention_combinenp[i][id] < partition:
                    attention_combinenp[i][id] = 0

#        partition = np.partition(attention_combinenp.flatten(), -13264)[-13264]
#        for id in np.ndindex(attention_combinenp.shape):
#            if attention_combinenp[id] < partition:
#                attention_combinenp[id] = 0

        attention_combine = torch.from_numpy(attention_combinenp).type(torch.float32)
#       torch.save(attention_combine, 'attention_combine_resize.pt')
        s5 = torch.count_nonzero(attention_combine)
        s6 = torch.count_nonzero(attention_combine, dim=1)
        print('attention_combine_countzero', s5)
        print('attention_combine_countzero_dim1', s6)

        one_vec = torch.ones_like(attention_combine)
        adj_resize = torch.where(attention_combine > 0, one_vec, 0*one_vec)
        
#        s14 = adj_resize[0:140, 0:140]
#        s15 = torch.count_nonzero(s14)
#        print("adj resize first 140", s15)


        s7 = adj * adj_resize
#        s11 = s7[0:140, 0:140]
        s8 = torch.count_nonzero(s7)
        print("s8", s8)
#        adj_resizenp = adj_resize.detach().cpu().numpy()      
#       adj_resizenp = adj_resizenp + sp.eye(adj_resizenp.shape[0])
#        adj_resizenp = self.normalize_adj(adj_resizenp)
#        adj_resize = torch.FloatTensor(np.array(adj_resize.todense()))
#       print('adj_pre', adj)
#        print('adj_resize', adj_resize)
        s9 = adj.transpose(0,1)
        print("check symmetry", ((adj == s9).all()))
#        print(torch.equal(adj_pre, adj_resize))
        torch.save(adj_resize, 'adj_resize_citeseer_004.pt')
        print('new adjacency matrix get')

        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def normalize_adj(mx):
        rowsum = np.array(mx.sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flattern()
        r_inv_sqrt[np.isinf(r_int_sqrt)] = 0
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
        return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
