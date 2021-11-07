import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        e = self._prepare_attentional_mechanism_input(Wh)

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

        print("attention tensor saved")
        torch.save(attention, 'attention.pt')

        attention2step = torch.mm(attention, attention.T)
#        attention2step = attention2step.fill_diagonal_(0)

        print("attention2step tensor saved")
        torch.save(attention2step, 'attention2step.pt')
        
        attention_combine = attention + attention2step

        print("attention_combine tensor saved")
        torch.save(attention_combine, 'attention_combine.pt')

        attention_combinenp = attention_combine.detach().cpu().numpy()
    
        partition = np.partition(attention_combinenp.flatten(), -13264)[-13264]
        for id in np.ndindex(attention_combinenp.shape):
            if attention_combinenp[id] < partition:
                attention_combinenp[id] = 0

        attention_combine = torch.from_numpy(attention_combinenp).type(torch.float32)
        torch.save(attention_combine, 'attention_combine_resize.pt')

        one_vec = torch.ones_like(attention_combine)
        adj_resize = torch.where(attention_combine > 0, one_vec, 0*one_vec)

        torch.save(adj_resize, 'adj_resize.pt')
        print('new adjacency matrix get')

        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

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