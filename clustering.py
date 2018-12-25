import json_to_list
from config import *

def split_words():
    poems,titles = json_to_list.json_to_list(file_name, poems_num)
    split_poems = {}
    t0 = time.time()
    for poem, title in zip(poems, titles):
        split_poem = jieba.lcut(poem)
        for stop_word in stop_words:
            while stop_word in split_poem:
                split_poem.remove(stop_word)
        split_poems[title] = split_poem
    print('分词用时{}秒'.format(time.time()-t0))
    return split_poems


def poems_to_corpus(poems):
    corpus = []
    for poem in poems:
        corpus.append(' '.join(poem))
    return corpus


def tf_idf(poems):
    t0 = time.time()
    corpus = poems_to_corpus(poems)
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    print('计算TF-IDF权重用时{}秒'.format(time.time()-t0))
    word = vectorizer.get_feature_names()
    print("词语特征数量：{}".format(len(word)))
    return tfidf


def cluster(poems, labels):
    value = np.unique(labels)
    clustered_poems = {}

    for cluster in value:
        clustered_poems[cluster] = []

    poems_list = list(poems.values())

    for index in range(len(labels)):
        clustered_poems[labels[index]].append(poems_list[index])

    return clustered_poems


def evaluate_kmeans(tfidf_weight, lower=2, upper=100, steps=5):
    index = 0
    max_score = 0
    for i in range(lower, upper, steps):
        kmeans = KMeans(n_clusters=i, random_state=1).fit(tfidf_weight)
        labels = kmeans.labels_
        score = metrics.silhouette_score(tfidf_weight, labels, metric='euclidean')
        print('簇数为{}时，轮廓系数得分为{}'.format(i, score))
        if score > max_score:
            max_score = score
            index = i
    return index, max_score

# def evaluate_ac(tfidf_weight, lower=2, upper=100, steps=5):
#     index = 0
#     max_score = 0
#     for i in range(lower, upper, steps):
#         clustering = AgglomerativeClustering(n_clusters=i, linkage='average').fit(tfidf_weight.toarray())
#         labels = clustering.labels_
#         score = metrics.silhouette_score(tfidf_weight, labels, metric='euclidean')
#         print('簇数为{}时，轮廓系数得分为{}'.format(i, score))
#         if score > max_score:
#             max_score = score
#             index = i
#     return index, max_score


def evaluate_ac(tfidf_weight, lower=2, upper=100, steps=5):
    index = 0
    max_score = 0
    for i in range(lower, upper, steps):
        clustering = AgglomerativeClustering(
            n_clusters=i, linkage='average').fit(tfidf_weight.toarray())
        labels = clustering.labels_
        score = metrics.silhouette_score(
            tfidf_weight, labels, metric='euclidean')
        print('簇数为{}时，轮廓系数得分为{}'.format(i, score))
        if score > max_score:
            max_score = score
            index = i
    return index, max_score



def Traditional2Simplified(sentence):
    sentence = Converter('zh-hans').convert(sentence)
    return sentence


def get_wordcloud(clustered_poems):
    for cluster in clustered_poems:
        words = ''
        # print(clustered_poems[cluster])
        for poems in clustered_poems[cluster]:
            word = ''
            poem = ' '.join(poems)
            word += poem
        words += word
        words = Traditional2Simplified(words)

        # print(type(words))
        # print(clustered_poems[cluster])
        wc = WordCloud(background_color="white", max_words=2000,
                       max_font_size=50, random_state=42,font_path= font_path)
        wc.generate(words)
        wc.to_file('wordcloud_cluster_{}_{}.png'.format(cluster,datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))

def demension_reduce(tfidf_weight):
    svd = TruncatedSVD(n_components = 100)
    svd.fit(tfidf_weight)
    return svd.components_



poems = split_words()
print('诗数：{}'.format(len(poems)))
# print(poems['日詩'])
# print(poems['登戎州江樓閑望'])
tfidf_weight = tf_idf(poems.values())
reduced = demension_reduce(tfidf_weight)

# AC

if evaluate:
    t0 = time.time()
    opt_cluster_num , opt_score = evaluate_ac(tfidf_weight)
    print('最优的簇数为{},此时轮廓系数得分为{}'.format(opt_cluster_num, opt_score))
    print('评估用时{}秒'.format(time.time()-t0))
    t0 = time.time()
    clustering = AgglomerativeClustering(n_clusters=opt_cluster_num, linkage='average')
    clustering.fit(reduced)
    print('聚类用时{}秒'.format(time.time()-t0))
else:
    t0 = time.time()
    clustering = AgglomerativeClustering(n_clusters=5, linkage='average')
    clustering.fit(reduced)
    print('聚类用时{}秒'.format(time.time()-t0))

clustered_poems = cluster(poems, clustering.labels_)
for i in range(len(clustered_poems)):
    print('第{}个簇有{}首古诗'.format(i, len(clustered_poems[i])),end=' ')
print()

# kmeans

# if evaluate:
#     t0 = time.time()
#     opt_cluster_num, opt_score = evaluate_kmeans(tfidf_weight)
#     print('最优的簇数为{},此时轮廓系数得分为{}'.format(opt_cluster_num, opt_score))
#     print('评估用时{}秒'.format(time.time() - t0))
#     t0 = time.time()
#     kmeans = KMeans(init='k-means++', n_clusters=opt_cluster_num)
#     kmeans.fit(tfidf_weight)
#     print('聚类用时{}秒'.format(time.time() - t0))
# else:
#     t0 = time.time()
#     kmeans = KMeans(init='k-means++', n_clusters=5)
#     kmeans.fit(tfidf_weight)
#     print('聚类用时{}秒'.format(time.time() - t0))
#
# clustered_poems = cluster(poems, kmeans.labels_)
# for i in range(len(clustered_poems)):
#     print('第{}个簇有{}首古诗'.format(i, len(clustered_poems[i])),end=' ')
# print()

# get wordcloud

get_wordcloud(clustered_poems)

