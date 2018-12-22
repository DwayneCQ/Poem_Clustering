import json
import os
import re
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import time
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances_argmin
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from langconv import *
import datetime

file_name = 'poet.song'
poems_num = 100000
stop_words = ['，', '。', '{', '}', '□', '(', ')', ]
font_path = '/Users/DwayneChen/Desktop/Poem_Clustering/JDJH.TTF'