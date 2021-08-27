import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import activation_getter


class DCNN2(nn.Module):
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
        self.units = self.args.units
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
        self.pool_size = filter(lambda x: x < self.dims, pool_size)
        self.drop_ratio = self.args.drop

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, self.dims)
        self.item_embeddings = nn.Embedding(num_items, self.dims)

        k1 = 1-(np.ones((self.L,self.L))+np.eye(self.L)*-1)
        k2 = 2*(np.ones((self.L,self.L))+np.eye(self.L)*-1)-1
        kernels = np.vstack([k1, k2])
        # DC
        self.convs = nn.Conv1d(2, 4, kernel_size=1, bias=False)
        self.convs.weight.data = torch.FloatTensor(kernels).reshape(self.L*2, self.L, 1) # nn.Parameter(K) #use nn.parameters
        self.convs.weight.requires_grad = False

        self.poolings = [nn.AvgPool1d(kernel_size=x, stride=x//2) for x in self.pool_size]

        self.dropout = nn.Dropout(self.drop_ratio)
        self.fc1 = nn.LazyLinear(self.units, bias=False)
        self.ac_conv = activation_getter[self.args.ac_conv]
        self.ac_fc = activation_getter[self.args.ac_fc]
        self.fc2 = nn.Linear(self.units, self.dims)

        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = nn.Embedding(num_items, self.dims+self.dims)
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
        item_embs = self.item_embeddings(seq_var)  # use unsqueeze() to get 4-D (:, 5, dim)
        user_emb = self.user_embeddings(user_var).squeeze(1)    # (:, 50)
        #tget_embs = self.user_embeddings(item_var).transpose(1, 2)   # (:, 50, 6)
        #print("item_embs.shape: " + str(item_embs.shape))
        #print("user_emb.shape: " + str(user_emb.shape))

        # Convolutional Layers
        _convs = self.convs(item_embs)
        #print(item_embs)
        #print(self.convs.weight.data)

        #print("_convs.shape: " + str(_convs.shape))
        _pools = [poolOp(_convs) for poolOp in self.poolings]
        #print("_pools.shape: " + str(_pools[0].shape))
        _channels = torch.cat([_convs] + _pools, dim=2)  # (:, 10, 49)
        #print("_channels.shape: " + str(_channels.shape))

        # fully-connected layer
        fc_in = torch.flatten(_channels, start_dim=1, end_dim=2) # (:, 490)
        fc_in = self.dropout(fc_in)
        z1 = self.ac_conv(self.fc1(fc_in))
        z2 = self.ac_fc(self.fc2(z1))
        x = torch.cat([z2, user_emb], dim=1) # (: 6, dim)
        #print("z2.shape: " + str(z2.shape))
        #print(z2)

        w2 = self.W2(item_var)
        b2 = self.b2(item_var)
        #print("w2.shape: " + str(w2.shape))
        #print("b2.shape: " + str(b2.shape))

        if for_pred:
            w2 = w2.squeeze()
            b2 = b2.squeeze()
            res = (x * w2).sum(1) + b2
        else:
            res = torch.baddbmm(b2, w2, x.unsqueeze(2)).squeeze()

        return res

