#! /usr/bin/python3
# -*- coding: utf-8 -*-


from collections import defaultdict
from glob import glob
import os
import re
import time
import json
import sys

from lxml import html
import numpy as np
import pandas as pd
import requests

# http://movie.naver.com/movie/point/af/list.nhn?st=mcode&sword=31150&target=after&page=2

BASEURL     = 'http://movie.naver.com/movie/point/af/list.nhn'
RATINGURL   = BASEURL + '?&page=%s'
MOVIEURL    = BASEURL + '?st=mcode&target=after&sword=%s&page=%s'

DATADIR     = 'data/ratings'
INDEXFILE   = 'data/index.txt'
TMPFILE     = 'data/movie2_ratings_all.txt'
RATINGSFILE = 'data/movie2_ratings.txt'
SEED        = 1234
SLEEP       = 600
NDOCS       = 400000


extract_nums = lambda s: re.search('\d+', s).group(0)
sanitize_str = lambda s: s.strip()

def write_txt(data,filename):

    file = open(filename, 'w')
    file.write(data)
    file.close()

    return

def write_json(data, filename):

    with open(filename, 'a') as f:
        json.dump(data, f)

    return

def read_json(filename):

    with open(filename,'rt',encoding='UTF8') as data_file:
        data = json.load(data_file)

    return data

def read_txt(filename):


    return



def parse_item(item):
    try:
        return {'review_id': item.xpath('./td[@class="ac num"]/text()')[0],     # num
                'rating': item.xpath('./td[@class="point"]/text()')[0],         # point
                'movie_id': extract_nums(item.xpath('./td[@class="title"]/a/@href')[0]),
                'review': sanitize_str(' '.join(item.xpath('./td[@class="title"]/text()'))),
                'author': item.xpath('./td[@class="num"]/a/text()')[0],
                'date': item.xpath('./td[@class="num"]/text()')[0]
        }
    except (IndexError, AttributeError) as e:
        print(e, item.xpath('.//text()'))
        return None
    except (AssertionError) as e:
        print(e, 'Sleep for %s' % SLEEP)
        time.sleep(SLEEP)
    except Exception as e:
        print(e, '음 여기까진 생각을 못했는데...')


def crawl_rating_page(url):
    resp = requests.get(url)
    root = html.fromstring(resp.text)
    items = root.xpath('//body//table[@class="list_netizen"]//tr')[1:]
    npages = max(map(int, ([0] + root.xpath('//div[@class="paging"]//a/span/text()'))))
    return list(filter(None, [parse_item(item) for item in items])), npages


def crawl_movie(movie_id):
    items = []
    for page_num in range(200):  # limit to 100 recent ratings per movie
        url = MOVIEURL % (movie_id, page_num + 1)
        page_items, npages = crawl_rating_page(url)
        items.extend(page_items)
        if len(items)==0:
            return []
        if page_num >= npages - 1:
            break
    if items:

        write_json(items, '%s/%s.json' % (DATADIR, movie_id))

        return items
    else:
        return []


def get_index(filename):
    movie_id, total = 31150, 0
    print(movie_id, total)
    return [movie_id, total]


def put_index(movie_id, total, filename):
    write_txt('%s,%s' % (movie_id, total), filename)

def merge_ratings():

    sub_space = lambda s: re.sub('\s+', ' ', s)
    write_row = lambda l, f: f.write('\t'.join(l) + '\n')

    filenames = glob('%s/*' % DATADIR)
    with open(TMPFILE, 'w') as f:
        write_row('id document'.split(), f)
        for filename in filenames:
            for review in read_json(filename):
                print(review)
                rating = int(review['rating'])
                write_row([review['review_id'], sub_space(review['review'])], f)
    print('Ratings merged to %s' % TMPFILE)

    df = pd.read_csv(TMPFILE, sep='\t', quoting=2)
    df = df.fillna('')
    np.random.seed(SEED)
    df = df.iloc[np.random.permutation(len(df))]
    df.to_csv(RATINGSFILE, sep='\t', index=False)
    print('Ratings written to %s' % RATINGSFILE)


if __name__=='__main__':
    movie_id, total = get_index(INDEXFILE)
    items = crawl_movie(movie_id)
    total += len(items)
    put_index(movie_id, total, INDEXFILE)
    print(MOVIEURL % (movie_id, 1), len(items), total)
    print(sys.stdin.encoding)
    merge_ratings()
