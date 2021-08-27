#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io, os, re, shutil, string
import click
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
import itertools

import torch
import torch.nn as nn

from interactions import Interactions
from utils import *


def test_conv1d():
    x2 = torch.rand(2, 2, 3, 2) #tf.random.normal(input_shape)

    kernel2 = [[1, 1],
               [1, -1],
               [1, 0],
               [0, 1]]
    kernel3 = [[[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]]]

    m = nn.Conv1d(2, 4, kernel_size=[1,1], bias=False)
    m.weight.data = torch.FloatTensor(kernel2).reshape(4, 2, 1, 1)
    output = m(x2)

    print(m.weight.data.shape)
    print(output.shape)

    print("------------")
    print(x2)
    print(output)

def test_pool():
    x2 = torch.rand(2, 2, 5, 2) #tf.random.normal(input_shape)

    m = nn.AvgPool2d(kernel_size=(3, 1), stride=(2, 1))
    output = m(x2)

    print(output.shape)

    print("------------")
    print(x2)
    print(output)

def test_sequence_data():
    # load dataset
    train = Interactions('datasets/ml1m/test/train.txt')
    # transform triplets to sequence representation
    train.to_sequence(5, 3)

    test = Interactions('datasets/ml1m/test/test.txt',
                        user_map=train.user_map,
                        item_map=train.item_map)

    print(train.test_sequences.user_ids)
    print(train.test_sequences.sequences)
    print(train.test_sequences.targets)
    print('--------------')
    print(train.sequences.user_ids)
    print(train.sequences.sequences)
    print(train.sequences.targets)

    df = test.tocsr()
    print(df.shape)
    for user_id, row in enumerate(df):
        print(user_id, row.indices)
        break

if __name__ == "__main__":
    #print("{}, {}, {}".format(date, dir, verbose))
    #test_conv1d()
    #test_pool()
    test_sequence_data()
