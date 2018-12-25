import json
import os
import re
import jieba
import numpy as np
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
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD

file_name = 'poet.song'
poems_num = 10000
stop_words = ['，', '。', '{', '}', '□', '(', ')', ]
font_path = '/Users/DwayneChen/Desktop/Poem_Clustering/JDJH.TTF'
evaluate = True
# opt_cluster_num = 67