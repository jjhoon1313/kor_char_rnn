# -*- coding: utf-8 -*-

from __future__ import print_function, division
import sys
import numpy as np
import re
import tensorflow as tf
import nltk
from konlpy.tag import Kkma, Mecab, Twitter
from pprint import pprint
from gensim.models import *
from matplotlib import font_manager, rc
from collections import namedtuple

font_fname = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)

twitter = Twitter()
kkma = Kkma()

def read_data(filename):
    with open(filename,'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data

def write_data(filename, data):
    with open(filename,'w') as f:
        f.write(data)
    return

def tokenize(doc):
    return['/'.join(t) for t in twitter.pos(doc, norm=True)]

# loading data
review_data = read_data('data/kowords_tag.txt')
review_data = [(tokenize(row[1]), row[2]) for row in review_data]

TaggedDocument = namedtuple('TaggedDocument','words tags')
tagged_review_docs = [TaggedDocument(d, [int(c)]) for d, c in review_data]

# 사전 구축
doc_vectorizer = doc2vec.Doc2Vec(size= 100, seed=1234, workers=6, min_count=1)
doc_vectorizer.build_vocab(tagged_review_docs)


# Train document vectors!
doc_vectorizer.train(tagged_review_docs,total_examples=343579, epochs=20, start_alpha=0.025, end_alpha=0.015)

doc_vectorizer.save('model/kowords_vec.model')


model_ko = Doc2Vec.load('model/kowords_vec.model')

ko_wv = model_ko.wv
ko_dv = model_ko.docvecs

pprint(ko_wv.word_vec('수학/Noun'))