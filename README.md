
# DC-PyTorch

A PyTorch implementation of Delay Convolutional Sequence Embedding Recommendation Model (DC) from the paper:

*not published*
Epoch 50 [162.8 s] loss=0.0419, map=0.1935, prec@1=0.3464, prec@5=0.2853, prec@10=0.2551, recall@1=0.0223, recall@5=0.0852, recall@10=0.1481, [69.7 s]

# Requirements
* Python 2 or 3
* [PyTorch v0.4+](https://github.com/pytorch/pytorch)
* Numpy
* SciPy

# Usage
1. Install required packages.
2. run <code>python train.py --model_type=dcnn</code>

# Configurations

#### Data

- Datasets are organized into 2 separate files: **_train.txt_** and **_test.txt_**

- Same to other data format for recommendation, each file contains a collection of triplets:

  > user item rating

  The only difference is the triplets are organized in *time order*.

- As the problem is Sequential Recommendation, the rating doesn't matter, so I convert them to all 1.

#### Model Args (in train_caser.py)

- <code>L</code>: length of sequence
- <code>T</code>: number of targets
- <code>d</code>: number of latent dimensions
- <code>drop_rate</code>: drop ratio when performing dropout
#- <code>nv</code>: number of vertical filters
#- <code>nh</code>: number of horizontal filters
#- <code>ac_conv</code>: activation function for convolution layer (i.e., phi_c in paper)
#- <code>ac_fc</code>: activation function for fully-connected layer (i.e., phi_a in paper)


# Citation

If you use this DCNN in your paper, please cite the paper:

```
@inproceedings{tang2018caser,
  title={Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding},
  author={Tang, Jiaxi and Wang, Ke},
  booktitle={ACM International Conference on Web Search and Data Mining},
  year={2018}
}
```

# Comments

1. This PyTorch version may get better performance than what the paper reports. 

   > When d=50, L=5, T=3, and set other arguments to default, after 20 epochs, mAP may get to 0.17 on the test set.

# Acknowledgment
This project is heavily build on project [caser_pytorch](https://github.com/graytowne/caser_pytorch), [GRU4Rec](https://github.com/hidasib/GRU4Rec), [CosRec](https://github.com/zzxslp/CosRec) and [Spotlight](https://github.com/maciejkula/spotlight), thanks all the guys for his great work.
