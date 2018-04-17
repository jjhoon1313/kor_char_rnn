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


BATCH_SIZE = 512

def read_data(filename):
    with open(filename,'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]  # header 제외
    return data

# set hyper-parameter
MAX_SEQ_LEN = 140

# INPUT STRING
input_docs = read_data('data/movie_ratings_all.txt')

twitter = Twitter()

def tokenize(doc):
    return['/'.join(t) for t in twitter.pos(doc)]

input_sentences = []
for section in input_docs:
    input_sentences.append(tokenize(section[1]))

# GET Sequence length


# Path for tf.summary.FileWriter and to store model checkpoints
FILEWRITER_PATH = 'tensorboard'
CHECKPOINT_PATH = 'tensorboard/checkpoints'


# Data Pre-Precessing
# word2vec model embedding

# Set Model
model = word2vec.Word2Vec.load('model/komorphs_word2vec3.model')

ko_wv = model.wv

test_wvs = []
test_seq_len = []
test_idx=0
for sentence in input_sentences:
    test_wv = []
    test_seq_len.append(len(sentence))
    if sentence != []:
        for word in sentence:
            if word != '':
                if word in ko_wv.vocab:
                    w_vec = ko_wv.word_vec(word)
                    test_wv.append(np.array(w_vec))

        if test_wv != []:
            test_wv = np.array(test_wv)
            test_wv = np.pad(test_wv,[(0,MAX_SEQ_LEN-test_wv.shape[0]),(0,0)], mode='constant', constant_values=0)
            test_wvs.append(test_wv)
        test_idx += 1

test_wvs = np.array(test_wvs)
test_seq_len = np.array(test_seq_len)

test_loop_count  = len(test_wvs) // BATCH_SIZE
print(test_loop_count)

save_path = os.path.join(CHECKPOINT_PATH,'kor-word2vec2-gruatt2-9_step-5553.meta')
model_path = os.path.join(CHECKPOINT_PATH,'kor-word2vec2-gruatt2-9_step-5553')

model = tf.train.import_meta_graph(save_path)

# session
config = tf.ConfigProto(gpu_options={'allow_growth':True})
sess = tf.InteractiveSession(config=config)

init_op = tf.global_variables_initializer()

sess.run(init_op)


model.restore(sess, model_path)
fake_label = np.zeros([BATCH_SIZE])

sum_result = 0
sum2_result = 0
for i in range(test_loop_count):
    t_start = time.time()
    offs = i * BATCH_SIZE

    batch_input = test_wvs[offs:offs + BATCH_SIZE]
    batch_input = np.reshape(batch_input, [BATCH_SIZE, 140, 100])
    batch_seqlen = test_seq_len[offs:offs + BATCH_SIZE]

    result = sess.run('result:0', feed_dict={'inputs:0':batch_input, 'inputs_seqlen:0':batch_seqlen, 'labels:0':fake_label, 'keep_prob:0':1.0})
    # sum_result += np.sum(result)
    sum2_result += np.sum(np.round(result))

print(sum_result)
# mean_result = sum_result/len(test_wvs)
mean2_result = sum2_result/len(test_wvs)

# print(mean_result)
print(mean2_result)

positive = mean2_result
negative = 1 - positive

print('토르:라그나로크 평점 평균')
print('긍정 : %.4f' % positive)
print('부정 : %.4f' % negative)