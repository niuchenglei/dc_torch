import argparse

from interactions import Interactions
from utils import *
from recommender import Recommender

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--train_root', type=str, default='datasets/ml1m/test/train.txt')
    parser.add_argument('--test_root', type=str, default='datasets/ml1m/test/test.txt')
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=3)
    # train arguments
    parser.add_argument('--n_iter', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--l1', type=float, default=0)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--use_cuda', type=str2bool, default=False)
    parser.add_argument('--loss', type=str, default='BPR', choices=['CE', 'BPR', 'OPT1', 'BPR-PD'])
    parser.add_argument('--model_type', type=str, default='dcnn', choices=['dcnn', 'dcnn_mlp', 'caser', 'cosrec', 'cosrec_base']) # CE/BPR/OPT1
    # model dependent arguments
    model_parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=50)
    parser.add_argument('--units', type=str, default='100,50')
    parser.add_argument('--pool_size', type=str, default='auto')
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--ac_conv', type=str, default='leaky_relu')
    parser.add_argument('--ac_fc', type=str, default='leaky_relu')

    config = parser.parse_args()
    print(config)

    # set seed
    set_seed(config.seed, cuda=config.use_cuda)

    if config.use_cuda:
        print('gpu_is_available: {0}'.format(torch.cuda.is_available()))
        gpu_count = torch.cuda.device_count()
        for idx in range(gpu_count):
            print('gpu {} is {}'.format(idx, torch.cuda.get_device_name(idx)))

    # load dataset
    train = Interactions(config.train_root)
    # transform triplets to sequence representation
    train.to_sequence(config.L, config.T)

    test = Interactions(config.test_root,
                        user_map=train.user_map,
                        item_map=train.item_map)

    # fit model
    model = Recommender(n_iter=config.n_iter,
                        batch_size=config.batch_size,
                        learning_rate=config.learning_rate,
                        l1=config.l1,
                        l2=config.l2,
                        neg_samples=config.neg_samples,
                        loss=config.loss,
                        model_args=config,
                        model_type=config.model_type,
                        use_cuda=config.use_cuda)

    model.fit(train, test, verbose=True)
