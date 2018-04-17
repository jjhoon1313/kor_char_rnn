#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np; np.random.seed(123)
import pandas as pd


ntrain = 280000
ntest = 40000

data = pd.read_csv('data/ratings.txt', sep='\t', quoting=3)
data = pd.DataFrame(np.random.permutation(data))
trn, tst = data[:ntrain], data[ntrain:ntrain+ntest]

header = 'id document label'.split()
trn.to_csv('../ratings_train.txt', sep='\t', index=False, header=header)
tst.to_csv('../ratings_test.txt', sep='\t', index=False, header=header)
