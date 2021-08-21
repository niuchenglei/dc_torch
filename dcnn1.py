import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import activation_getter


class DCNNMLP(nn.Module):
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
        super(DCNNMLP, self).__init__()
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
                if v < 5:
                    break
                pool_size.append(v)
        else:
            pool_size = [int(x) for x in self.args.pool_size.split(',')]
        self.pool_size = filter(lambda x: x < self.dims, pool_size)
        self.drop_ratio = self.args.drop

        # user and item embeddings
        #self.user_embeddings = nn.Embedding(num_users, self.dims)
        self.item_embeddings = nn.Embedding(num_items, self.dims)

        # DC
        self.convs = nn.Conv1d(2, 4, kernel_size=[1,1], bias=False)
        self.convs.weight.data = torch.FloatTensor([[1, 1], [1, -1], [1, 0], [0, 1]]).reshape(4, 2, 1, 1) # nn.Parameter(K) #use nn.parameters
        self.convs.weight.requires_grad = False

        self.poolings = [nn.AvgPool2d(kernel_size=(x, 1), stride=(x//2, 1)) for x in self.pool_size]

        self.dropout = nn.Dropout(self.drop_ratio)
        self.fc1 = nn.LazyLinear(self.units)
        #self.fc2 = nn.Linear(self.units, 1)
        self.ac_fc = activation_getter[self.args.ac_fc]

        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = nn.Embedding(num_items, self.units)
        self.b2 = nn.Embedding(num_items, 1)

        # weight initialization
        #self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
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
        item_embs = self.item_embeddings(seq_var).transpose(1, 2)       # (b, embed, L)
        #user_emb = self.user_embeddings(user_var)                       # (:, 1, 50)
        tget_embs = self.item_embeddings(item_var).transpose(1, 2)      # (:, 50, 6)
        target_size = tget_embs.shape[2]

        # Convolutional Layers
        item_emb_sum = torch.sum(item_embs, dim=2, keepdim=True)        # (:, 50, 1)
        item_emb_sum_align = item_emb_sum.repeat(1, 1, target_size)     # (:, dim, 6)

        emb_pair = torch.cat([tget_embs.unsqueeze(1), item_emb_sum_align.unsqueeze(1)], dim=1)  # (:, 2, dim, channels)

        _convs = self.convs(emb_pair)                                   # (:, 4, dim, 6)
        _dots = torch.mul(tget_embs, item_emb_sum_align).unsqueeze(1)   # (:, 1, dim, 6)
        _channels = torch.cat([_convs, _dots], dim=1)                   # (:, 5, dim, 6)

        _pools = [poolOp(_channels) for poolOp in self.poolings]
        _flatten = torch.cat([_channels] + _pools, dim=2)

        # fully-connected layer
        fc_in = torch.flatten(_flatten, start_dim=1, end_dim=2).transpose(1, 2)     # (:, 6, ?)
        x = self.ac_fc(self.fc1(fc_in))                                             # (:, 6, 100)
        #z2 = self.fc2(z1)
        #z2 = torch.cat([self.fc2(z1), user_emb], dim=1)                             # (: 6, dim)
        #print("x.shape: " + str(x.shape))

        w2 = self.W2(item_var)
        b2 = self.b2(item_var)

        if for_pred:
            w2 = w2.squeeze() # 3416, 100
            b2 = b2.squeeze() # 3416
            x = x.squeeze()
            res = torch.diagonal(torch.matmul(w2, x.permute(1, 0)), dim1=0, dim2=1) + b2
        else:
            res = torch.diagonal(torch.matmul(w2, x.permute(0, 2, 1)), dim1=1, dim2=2) + b2.squeeze()

        return res
