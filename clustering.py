from config import *
import json_to_list
from S_Dbw import *

def split_words():
    poems,titles = json_to_list.json_to_list(file_name, poems_num)
    split_poems = {}
    t0 = time.time()
    for poem, title in zip(poems, titles):
        split_poem = []
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

def evaluate():
    pass

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
        wc.to_file('wordcloud_cluster_{}.png'.format(cluster))


poems = split_words()
tfidf_weight = tf_idf(poems.values())
# print(poems_vec)
t0 = time.time()
index = []
scores = []
kmeans = KMeans(init='k-means++', n_clusters=10)
kmeans.fit(tfidf_weight)
print('聚类用时{}秒'.format(time.time()-t0))

clustered_poems = cluster(poems, kmeans.labels_)
for i in range(len(clustered_poems)):
    print('第{}个簇有{}首古诗'.format(i, len(clustered_poems[i])),end=' ')

get_wordcloud(clustered_poems)



# t0 = time.time()
# for i in range(kmeans.cluster_centers_.shape[0]):
#     print('第{}类'.format(i))
    # k_means_cluster_centers = kmeans.cluster_centers_
    # k_means_labels = pairwise_distances_argmin(tfidf_weight, k_means_cluster_centers)
    # score = sdbw.S_Dbw(np.asmatrix(tfidf_weight), k_means_labels , k_means_labels).S_Dbw_result()

# print('评估用时{}秒'.format(time.time()-t0))
# plt.plot(index, scores)
# plt.show()



