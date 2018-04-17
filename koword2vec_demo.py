# -*- coding: utf-8 -*-

from __future__ import print_function, division
import sys
import numpy as np
from pprint import pprint
from gensim.models import *
from matplotlib import font_manager, rc

font_fname = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)

model_ko = Doc2Vec.load('model/ko_doc2vec.bin')

ko_wv = model_ko.wv

pprint(ko_wv.word_vec('유비'))