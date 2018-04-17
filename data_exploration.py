# -*- coding: utf-8 -*-

from __future__ import print_function, division
import sys
import time
import numpy as np
import math
import re
import tensorflow as tf
import nltk
from konlpy.tag import Kkma, Twitter
from pprint import pprint
from matplotlib import font_manager, rc
import pandas as pd

font_fname = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)

# def read_data(filename):
#     with open(filename,'r') as f:
#         data = [line.split(' ') for line in f.readlines()]
#     return data


def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]   # header 제외
    return data

#
# def read_data2(filename):
#     with open(filename,'r') as f:
#         data = [line.split('\n') for line in f.readlines()]
#         data = data[1:]
#         newData = []
#         for line in data:
#             newData += [line[0].split('\t')]
#     return newData


kkma = Kkma()
twitter = Twitter()

# print(pos_review[:10])
#
# max_seq = 0
#
# print(len(pos_review[0][0].split(" "))-1)
#
# for tag_sentence in [pos_review[x][0] for x in range(len(pos_review))]:
#     if max_seq < len(tag_sentence.split(" "))-1:
#         max_seq = len(tag_sentence.split(" "))-1
#
# print(max_seq)

def tokenize(doc):
    return['/'.join(t) for t in twitter.pos(doc)]

# train_data = read_data2('data/ratings_train.txt')
# print(train_data[:2])
# test_data = read_data2('data/ratings_test.txt')
# ko_words = read_data('data/komorphs.txt')
#
# train_data = read_data2('data/ratings_train.txt')
# test_data = read_data2('data/ratings_test.txt')
#
# print(train_data[:10])
# train_data = pd.read_csv('data/ratings_train.txt', sep='\t', quoting=3)
# test_data = pd.read_csv('data/ratings_test.txt', sep='\t', quoting=3)
#
# # train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
# # test_docs = [(tokenize(row[1]), row[2]) for row in test_data]
# kowords_docs = [(tokenize(row[1]), row[2]) for row in ko_words]
#
# # tokens = [t for d in train_docs for t in d[0]]
# # tokens = [t for d in kowords_docs for t in d[0]]
#
# with open('data/train_input.txt','w') as f:
#     for i in range(len(train_data)):
#         doc = train_data.get_value(i, 'document')
#         tokened = tokenize(doc)
#         for token in tokened:
#             f.write(token+' ')
#         f.write('\t'+str(train_data.get_value(i,'label'))+'\n')

# with open('data/train_input.txt','w') as f:
#     for data in train_data:
#         tokened = tokenize(data[1])
#         for token in tokened:
#             f.write(token+' ')
#         f.write('\t'+data[2]+'\n')
#
# with open('data/train_test.txt', 'w') as f:
#     for data in test_data:
#         tokened = tokenize(data[1])
#         for token in tokened:
#             f.write(token + ' ')
#         f.write('\t' + data[2] + '\n')

# with open('data/test_input.txt','w') as f:
#     for i in range(len(test_data)-1):
#         doc = test_data.get_value(i, 'document')
#         tokened = tokenize(doc)
#         for token in tokened:
#             f.write(token+' ')
#         f.write('\t'+str(test_data.get_value(i,'label'))+'\n')

# with open('data/komorphs_tag.txt','w') as f:
#     for row in ko_words:
#         for word in row:
#             tokened = tokenize(word)
#             for token in tokened:
#                 f.write(token+' ')
#         f.write('\n')

# data = read_data('data/ratings_train.txt')
test_data = read_data('data/ratings_test.txt')

# train_docs = [(tokenize(row[1]), row[2]) for row in data]
test_docs = [(tokenize(row[1]), row[2]) for row in test_data]

# tokens = [t for d in train_docs for t in d[0]]
tokens2 = [t for d in test_docs for t in d[0]]
# tokens = [t for d in kowords_docs for t in d[0]]

# text = nltk.Text(tokens, name='aaa')
text2 = nltk.Text(tokens2, name='bbb')

# text.plot(50)
text2.plot(50)