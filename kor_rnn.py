# -*- coding: utf-8 -*-

from __future__ import print_function, division
import sys
import os
import time
import numpy as np
import re
from gensim.models import word2vec, doc2vec
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import dynamic_rnn as rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn

from attention import attention
from utils import *


# READ TRAIN/VALID DATA
def read_data2(filename):
    with open(filename,'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data

input_train = read_data2('data/train_input2.txt')
input_test = read_data2('data/test_input2.txt')


# GET Max Sequence length
max_seq = 0

for tag_sentence in [input_train[x][0] for x in range(len(input_train))]:
    if max_seq < len(tag_sentence.split(" "))-1:
        max_seq = len(tag_sentence.split(" "))-1

for tag_sentence in [input_test[x][0] for x in range(len(input_test))]:
    if max_seq < len(tag_sentence.split(" ")) - 1:
        max_seq = len(tag_sentence.split(" ")) - 1


# Set Hyper-parameters

INPUT_UNITS = 100
NUM_HIDDEN_UNITS = 150
ATTENTION_SIZE = 50
KEEP_PROB = 0.8
MAX_SEQ_LEN = max_seq
print(max_seq)
BATCH_SIZE = 512
NUM_EPOCH = 20
DELTA = 0.5


# Path for tf.summary.FileWriter and to store model checkpoints
FILEWRITER_PATH = 'tensorboard'
CHECKPOINT_PATH = 'tensorboard/checkpoints'

# Recover all weight variables from the last checkpoint
RECOVER_CKPT = False

# Create parent path if it doesn't exist
if not os.path.isdir(FILEWRITER_PATH):
    os.makedirs(FILEWRITER_PATH)

if not os.path.isdir(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)

# print(len(input_test))
# 43579

# print(len(input_train))
# 300000

# Data Pre-Precessing
# word2vec model embedding

# train_label = np.array([int(input_train[x][1]) for x in range(len(input_train))])
# test_label = np.array([int(input_test[x][1]) for x in range(len(input_test))])
train_label = []
test_label = []

model = word2vec.Word2Vec.load('model/komorphs_word2vec3.model')
ko_wv = model.wv
train_inputs = np.array([[input_train[x][0]] for x in range(len(input_train))])
test_inputs = np.array([[input_test[x][0]] for x in range(len(input_test))])


train_wvs = []
train_seq_len = []
train_idx = 0
for sentence in train_inputs:
    words = sentence[0].split(" ")[:-1]
    train_wv = []
    train_seq_len.append(len(words))
    if words != []:
        for word in words:
            if word != '':
                if word in ko_wv.vocab:
                    w_vec = ko_wv.word_vec(word)
                    train_wv.append(np.array(w_vec))

        if train_wv != []:
            train_wv = np.array(train_wv)
            train_wv = np.pad(train_wv, [(0,MAX_SEQ_LEN-train_wv.shape[0]),(0,0)],mode='constant',constant_values=0)
            train_wvs.append(train_wv)
            train_label.append(input_train[train_idx][1])
    train_idx += 1


test_wvs = []
test_seq_len = []
test_idx=0
for sentence in test_inputs:
    words = sentence[0].split(" ")[:-1]
    test_wv = []
    test_seq_len.append(len(words))
    if words != []:
        for word in words:
            if word != '':
                if word in ko_wv.vocab:
                    w_vec = ko_wv.word_vec(word)
                    test_wv.append(np.array(w_vec))

        if test_wv != []:
            test_wv = np.array(test_wv)
            test_wv = np.pad(test_wv,[(0,MAX_SEQ_LEN-test_wv.shape[0]),(0,0)], mode='constant', constant_values=0)
            test_wvs.append(test_wv)
            test_label.append(input_test[test_idx][1])
        test_idx += 1

train_wvs = np.array(train_wvs)
test_wvs = np.array(test_wvs)

train_seq_len = np.array(train_seq_len)
test_seq_len = np.array(test_seq_len)

train_loop_count = len(train_wvs) // BATCH_SIZE
test_loop_count  = len(test_wvs) // BATCH_SIZE

print('trainig loop count : %d' % train_loop_count)
print('test loop count : %d' % test_loop_count)

# Set Model
# tf.graph, placeholders
tf.reset_default_graph()

inputs_ = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_SEQ_LEN, INPUT_UNITS], name='inputs')
inputs_seqlen = tf.placeholder(tf.int32, [BATCH_SIZE], name='inputs_seqlen')
labels_ = tf.placeholder(tf.float32, [BATCH_SIZE], name='labels')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')


# Set GRU-Attention model

gru_cell = GRUCell(NUM_HIDDEN_UNITS)

# RNN 모델 구성
rnn_outputs, states = bi_rnn(gru_cell, gru_cell, inputs=inputs_, sequence_length=inputs_seqlen, dtype=tf.float32)

# Attention Layer
attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)

# Dropout
drop = tf.nn.dropout(attention_output, keep_prob, name='do1')

# Fully Connected Layer(FCN)
W = tf.Variable(tf.truncated_normal([drop.get_shape()[1].value, 1], stddev=0.1), name='w1')
b = tf.Variable(tf.constant(0., shape=[1]), name='b1')
y_hat = tf.nn.xw_plus_b(drop, W, b, name='y_hat')
y_hat = tf.squeeze(y_hat)

result = tf.round(tf.sigmoid(y_hat, name='result'))


# Cross-entropy loss and optimizer initialization
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=labels_))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Accuracy metric
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(y_hat)), labels_), tf.float32))

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Train,,,
# session
config = tf.ConfigProto(gpu_options={'allow_growth':True})
sess = tf.InteractiveSession(config=config)

tf.global_variables_initializer().run()

# (optional) load model weights
if RECOVER_CKPT:
    latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_PATH)
    print("Loading the last checkpoint: " + latest_ckpt)
    saver.restore(sess, latest_ckpt)
    last_epoch = int(latest_ckpt.replace('_', '*').replace('-', '*').split('*')[3])
else:
    last_epoch = 0

# tf.summary
train_writer = tf.summary.FileWriter('logdir/train2', graph=tf.get_default_graph())
test_writer  = tf.summary.FileWriter('logdir/test2', graph=tf.get_default_graph())

def train(
        train_inputs,
        train_seqlens,
        train_labels,
        test_inputs,
        test_seqlens,
        test_labels,
        max_epochs,
        train_writer=None,
        test_writer=None):
    step = 0
    print('Start Learning...')
    for ep in range(last_epoch, max_epochs):

        loss_train = 0
        accuracy_train = 0
        accuracy_test = 0

        print("epoch: {}\t".format(ep+1), end="")

        train_elapsed = []
        print('Start Training..')
        for i in range(train_loop_count):
            t_start = time.time()

            offs = i * BATCH_SIZE

            batch_input = train_inputs[offs:offs + BATCH_SIZE]
            batch_input = np.reshape(batch_input, [BATCH_SIZE, MAX_SEQ_LEN, INPUT_UNITS])
            batch_seqlen = train_seqlens[offs:offs+BATCH_SIZE]
            batch_label = train_labels[offs:offs + BATCH_SIZE]

            _, loss_tr, acc = sess.run([optimizer, loss, accuracy], feed_dict={inputs_: batch_input, inputs_seqlen: batch_seqlen, labels_: batch_label, keep_prob: KEEP_PROB})

            accuracy_train += acc
            loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
            t_elapsed = time.time() - t_start
            train_elapsed.append(t_elapsed)

            step += 1

            if train_writer:
                summary = tf.Summary( value=[tf.Summary.Value(tag='train_accuracy', simple_value=acc),
                                             tf.Summary.Value(tag='loss', simple_value=loss_train)])
                train_writer.add_summary(summary, global_step=step)

        accuracy_train /= train_loop_count
        print(('[trn] ep {:d}, step {:d}, loss {:f}, accu {:f}, sec/iter {:f}').format(ep + 1, step, loss_train, accuracy_train, np.sum(train_elapsed)))

        # Testing

        test_elapsed = []
        print('validation start...')

        for i in range(test_loop_count):
            t_start = time.time()
            offs = i * BATCH_SIZE

            batch_input = test_inputs[offs:offs + BATCH_SIZE]
            batch_input = np.reshape(batch_input, [BATCH_SIZE, MAX_SEQ_LEN, INPUT_UNITS])
            batch_seqlen = test_seqlens[offs:offs+BATCH_SIZE]
            batch_label = test_labels[offs:offs + BATCH_SIZE]

            acc = sess.run(accuracy, feed_dict={inputs_: batch_input, inputs_seqlen : batch_seqlen, labels_: batch_label, keep_prob: 1.0})

            accuracy_test += acc
            t_elapsed = time.time() - t_start
            test_elapsed.append(t_elapsed)

            step += 1

        accuracy_test /= test_loop_count

        if test_writer:
            summary = tf.Summary(value=[tf.Summary.Value(tag='test_accuracy', simple_value=accuracy_test)])
            test_writer.add_summary(summary, global_step=step)

        print(('[tst] ep {:d}, step {:d}, accu {:f}, sec/iter {:f}').format(ep + 1, step, accuracy_test, np.sum(test_elapsed)))

        # save checkpoint of the model at each epoch
        print("Saving checkpoint of model...")
        checkpoint_name = os.path.join(CHECKPOINT_PATH, 'kor-word2vec2-gruatt2-' + str(ep + 1) + '_step')
        save_path = saver.save(sess, checkpoint_name, global_step=step)
        print("Epoch: %d, Model checkpoint saved at %s" % (ep + 1, checkpoint_name + '-(#global_step)'))

# train
tf.get_default_graph().finalize()

train(train_wvs, train_seq_len, train_label, test_wvs, test_seq_len, test_label, NUM_EPOCH, train_writer, test_writer)