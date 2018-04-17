# -*- coding: utf-8 -*-

from __future__ import print_function, division
import sys
import os
import time
import numpy as np
import re
from gensim.models import word2vec, doc2vec
from konlpy.tag import Twitter
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import dynamic_rnn as rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn

from attention import attention
from utils import *

# set hyper-parameter
MAX_SEQ_LEN = 140

# INPUT STRING
input_sentence = input('Please enter the sentences :')

twitter = Twitter()

def tokenize(doc):
    return['/'.join(t) for t in twitter.pos(doc)]

input_words = tokenize(input_sentence)

# GET Sequence length


# Path for tf.summary.FileWriter and to store model checkpoints
FILEWRITER_PATH = 'tensorboard'
CHECKPOINT_PATH = 'tensorboard/checkpoints'


# Data Pre-Precessing
# word2vec model embedding

# Set Model
model = word2vec.Word2Vec.load('model/komorphs_word2vec3.model')

ko_wv = model.wv

test_wv = []
for word in input_words:
    if word in ko_wv.vocab:
        w_vec = ko_wv.word_vec(word)
        test_wv.append(np.array(w_vec))

test_wv = np.array(test_wv)
test_wv = np.pad(test_wv, [(0,MAX_SEQ_LEN-test_wv.shape[0]),(0,0)],mode='constant',constant_values=0)

test_wvs = np.array([test_wv] * 512)

test_wv = np.array(test_wv)
test_seq_len = np.array([len(input_words)] * 512)

save_path = os.path.join(CHECKPOINT_PATH,'kor-word2vec2-gruatt2-9_step-5553.meta')
model_path = os.path.join(CHECKPOINT_PATH,'kor-word2vec2-gruatt2-9_step-5553')

model = tf.train.import_meta_graph(save_path)

# session
config = tf.ConfigProto(gpu_options={'allow_growth':True})
sess = tf.InteractiveSession(config=config)

init_op = tf.global_variables_initializer()

sess.run(init_op)

fake_label = np.zeros([512])
model.restore(sess, model_path)

result, alphas, att_outputs, weight, bias = sess.run(['y_hat:0', 'alphas:0', 'output:0', 'w1:0', 'b1:0'], feed_dict={'inputs:0':test_wvs, 'inputs_seqlen:0':test_seq_len, 'labels:0':fake_label, 'keep_prob:0':1.0})

print(input_sentence)
print(input_words)
print(alphas[0][:len(input_words)])
print(result[0])

def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

positive = sigmoid(result[0])
negative = 1 - positive

print('%s' % input_sentence)
print('긍정 : %.4f' % positive)
print('부정 : %.4f' % negative)