import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import activation_getter


class DCNN(nn.Module):
    """
    Convolutional Sequence Embedding Recommendation Model (DCNN)[1].

    [1] Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding, Jiaxi Tang and Ke Wang , WSDM '18

    Parameters
    ----------

    num_users: int,
        Number of users.
    num_items: int,
        Number of items.
    model_args: args,
        Model-related arguments, like latent dimensions.
    """

    def __init__(self, num_users, num_items, model_args):
        super(DCNN, self).__init__()
        self.args = model_args

        # init args
        self.L = self.args.L
        self.dims = self.args.d
        self.units = [int(x) for x in self.args.units.split(',')]
        pool_size = []
        if self.args.pool_size == 'auto':
            pool_size.append(self.dims//2)
            while True:
                v = pool_size[-1] // 2
                if v < 3:
                    break
                pool_size.append(v)
        else:
            pool_size = [int(x) for x in self.args.pool_size.split(',')]
        self.pool_size = list(filter(lambda x: x < self.dims, pool_size))
        self.drop_ratio = self.args.drop

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, self.dims)
        self.item_embeddings = nn.Embedding(num_items, self.dims)

        self.triu_indices = torch.triu_indices(self.L, self.L, offset=0)
        permutation_size = self.triu_indices[0].shape[0]

        self.convs = nn.Conv1d(2, 4, kernel_size=[1,1], bias=False)
        self.convs.weight.data = torch.FloatTensor([[1, 1], [1, -1], [1, 0], [0, 1]]).reshape(4, 2, 1, 1)
        self.convs.weight.requires_grad = False

        self.poolings = [nn.MaxPool2d(kernel_size=(1,x), stride=(1,x//2)) for x in self.pool_size]
        self.dc_bn = nn.BatchNorm1d(permutation_size)

        self.dropout = nn.Dropout(self.drop_ratio)
        self.ac_conv = activation_getter[self.args.ac_conv]
        self.ac_fc = activation_getter[self.args.ac_fc]
        self.fc_layers = nn.ModuleList([nn.LazyLinear(self.units[idx]) for idx in range(0, len(self.units))])
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(self.units[idx] if idx>0 else permutation_size) for idx in range(0, len(self.units))])

        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = nn.Embedding(num_items, self.dims+self.units[-1])
        self.b2 = nn.Embedding(num_items, 1)

        # weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

    def forward(self, seq_var, user_var, item_var, for_pred=False):
        """
        The forward propagation used to get recommendation scores, given
        triplet (user, sequence, targets).

        Parameters
        ----------

        seq_var: torch.FloatTensor with size [batch_size, max_sequence_length]
            a batch of sequence
        user_var: torch.LongTensor with size [batch_size]
            a batch of user
        item_var: torch.LongTensor with size [batch_size]
            a batch of items
        for_pred: boolean, optional
            Train or Prediction. Set to True when evaluation.
        """

        # Embedding Look-up
        item_embs = self.item_embeddings(seq_var)               # (b, L, embed)(b, 5, 50)
        user_emb = self.user_embeddings(user_var)               # (b, 1, embed)

        # cast all item embeddings pairs against each other
        item_i = torch.unsqueeze(item_embs, 1)                                      # (b, 1, 5, embed)
        item_i = item_i.repeat(1, self.L, 1, 1).unsqueeze(1)                        # (b, 1, 5, 5, embed)
        item_j = torch.unsqueeze(item_embs, 2)                                      # (b, 5, 1, embed)
        item_j = item_j.repeat(1, 1, self.L, 1).unsqueeze(1)                        # (b, 1, 5, 5, embed)
        all_embs = torch.cat([item_i, item_j], 1)                                   # (b, 2, 5, 5, embed)
        perm_embs = all_embs[:, :, self.triu_indices[0], self.triu_indices[1], :]   # (b, 2, (L*(L-1))/2, embed)
        
        # Convolutional Layers
        _convs = self.convs(perm_embs)     
        _dots = torch.unsqueeze(perm_embs[:, 0, :, :] * perm_embs[:, 1, :, :], 1)   # (b, 1, (L*(L-1))/2, embed)
        _channels = torch.cat([_convs, _dots], dim=1)                               # (b, 5, (L*(L-1))/2, embed)

        _pools = [poolOp(_channels) for poolOp in self.poolings]                    # (b, 5, (L*(L-1))/2, 3/5/7/48)
        _dcout = torch.cat([_channels] + _pools, dim=3)
        _dcout = _dcout.permute(0, 2, 1, 3)                                         # (b, (L*(L-1))/2, 5, dim_pool)  // dim_pool = d + d//2 + d//4 + ...
        _dcout = torch.flatten(_dcout, start_dim=2)                                 # (b,  (L*(L-1))/2, 5*dim_pool)
        _dcout = self.dc_bn(_dcout)
        
        # delay convoluational neural network layer
        _dcnn = self.fc_layers[0](self.dropout(_dcout))
        _dcnn = self.ac_conv(self.bn_layers[0](_dcnn))
        z = torch.flatten(_dcnn, start_dim=1)
        
        # full connection layers
        for idx in range(1, len(self.fc_layers)):
            linear = self.fc_layers[idx]
            z = self.ac_fc(self.bn_layers[idx](linear(self.dropout(z))))

        x = torch.cat([z, user_emb.squeeze(1)], dim=1)                              # (: dim+units)

        w2 = self.W2(item_var)
        b2 = self.b2(item_var)

        if for_pred:
            w2 = w2.squeeze()
            b2 = b2.squeeze()
            res = (x * w2).sum(1) + b2
        else:
            res = torch.baddbmm(b2, w2, x.unsqueeze(2)).squeeze()

        return res

