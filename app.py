import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import brown
from nltk.cluster.util import cosine_distance
from nltk.tokenize import TreebankWordTokenizer
from operator import itemgetter
from function import TextCleaner
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from gensim.models import Word2Vec
from multiprocessing import Pool

nltk.download('brown')

st.sidebar.subheader("Dataset parameter")
banyak_data = st.sidebar.slider("Berapa Dataset", 0, len(brown.fileids()), 10)
dataset = st.sidebar.selectbox("Choose Brown Dataset?", brown.fileids()[:banyak_data])
sentences = brown.sents(dataset)
list_sentences = [' '.join(sent) for sent in sentences]
st.header("Summarization Corpus Brown")
st.dataframe(list_sentences)

# Function
def pagerank(M, eps=1.0e-8, d=0.85):
    N = M.shape[1]
    v = np.random.rand(N, 1)
    v = v / np.linalg.norm(v, 1)
    last_v = np.ones((N, 1), dtype=np.float32) * np.inf
    M_hat = (d * M) + (((1 - d) / N) * np.ones((N, N), dtype=np.float32))
    
    while np.linalg.norm(v - last_v, 2) > eps:
        last_v = v
        v = np.matmul(M_hat, v)
    return v
  
def sentence_similarity(sent1, sent2):
    text_cleaner = TextCleaner()
    
    sent1 = text_cleaner.clean_up(sent1)
    sent2 = text_cleaner.clean_up(sent2)
    
    all_words = list(set(sent1 + sent2))
    
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    for w in sent1:
        vector1[all_words.index(w)] += 1
    
    for w in sent2:
        vector2[all_words.index(w)] += 1
    
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences):
    S = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            else:
                S[i][j] = sentence_similarity(sentences[i], sentences[j])
    
    for i in range(len(S)):
        S[i] /= S[i].sum()
    return S

def build_lexicon(corpus):
    lexicon = set()
    for doc in corpus:
        lexicon.update([word for word in doc])
    return lexicon

# word embedding
def word_embedding(sen):
    embeded = 0
    vocabulary = build_lexicon(sentences)
    word_list = [word for word in vocabulary]
    for i in range(len(word_list)):
        if ((word_list[i] in word2vec_model.wv.index2word) == True):
            embeded = embeded + word2vec_model.wv.get_vector(word_list[i])
        else:
            embeded = embeded + unknown_embedd
    return embeded

st.sidebar.subheader("Method Parameter")
genre = st.sidebar.radio("What's your Method",('TextRank', 'wordembedRank', 'wordembedCluster', 'compareMethod'))
if genre == 'TextRank':
    st.subheader("Sentence Ranking")
    col1, col2 = st.beta_columns([3, 1])
    S = build_similarity_matrix(sentences)
    col1.write(S)
    sentence_ranks = pagerank(S)
    col2.write(sentence_ranks)
    
    # Load Word Sense Disambiguation 
    st.subheader("Index Sentence Ranking")
    col3, col4 = st.beta_columns([3, 1])
    ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
    col3.dataframe(ranked_sentence_indexes)
    st.sidebar.subheader("Summary Parameter")
    SUMMARY_SIZE = st.sidebar.slider("Berapa Jumlah Size?", 0, 10, 5)
    selected_sentences = sorted(ranked_sentence_indexes[:SUMMARY_SIZE])
    col4.dataframe(selected_sentences)

    st.subheader("Summary Result")
    summary = itemgetter(*selected_sentences)(sentences)
    for sent in summary:
        st.write(' '.join(sent))

elif genre == 'wordembedRank':
    st.subheader("Sentence Ranking based on Disamiguation")
    # Load Word Sense Disambiguation 
    st.sidebar.subheader("Word2vec Parameter")
    size_value = st.sidebar.slider("Berapa size?", 0, 200, len(sentences))
    mode_value = st.sidebar.selectbox("Pilih Mode", [1, 0])
    window_value = st.sidebar.slider("WIndows Size?", 0, 10, 3)
    iteration_value = st.sidebar.slider("iteration size?", 0, 100, 10) 
    word2vec_model = Word2Vec(sentences = sentences, size = size_value, sg = mode_value, window = window_value, min_count = 1, iter = iteration_value, workers = Pool()._processes)
    word2vec_model.init_sims(replace = True)

    col1, col2 = st.beta_columns([3, 1])
    st.subheader("Sentence Ranking")
    vector = [word_embedding(sentences[i]) for i in range(len(sentences))]
    vector_df = pd.DataFrame(vector)
    col1.write(vector_df)
    sentence_ranks = pagerank(vector_df)
    col2.write(sentence_ranks)
    
    # Load Word Sense Disambiguation 
    st.subheader("Index Sentence Ranking")
    col3, col4 = st.beta_columns([3, 1])
    ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
    col3.dataframe(ranked_sentence_indexes)
    st.sidebar.subheader("Summary Parameter")
    SUMMARY_SIZE = st.sidebar.slider("Berapa Jumlah Size?", 0, 10, 5)
    selected_sentences = sorted(ranked_sentence_indexes[:SUMMARY_SIZE])
    col4.dataframe(selected_sentences)

    st.subheader("Summary Result")
    summary = itemgetter(*selected_sentences)(sentences)
    hasilSummary = [' '.join(sent) for sent in summary]
    st.write(hasilSummary)
#     for sent in summary:
#         st.write(' '.join(sent))
elif genre == 'wordembedCluster':
    # Load word2vec pretrained
    st.sidebar.subheader("Word2vec Parameter")
    size_value = st.sidebar.slider("Berapa size?", 0, 200, len(sentences))
    mode_value = st.sidebar.selectbox("Pilih Mode", [1, 0])
    window_value = st.sidebar.slider("WIndows Size?", 0, 10, 3)
    iteration_value = st.sidebar.slider("iteration size?", 0, 100, 10) 
    word2vec_model = Word2Vec(sentences = sentences, size = size_value, sg = mode_value, window = window_value, min_count = 1, iter = iteration_value, workers = Pool()._processes)
    word2vec_model.init_sims(replace = True)
    embedd_vectors = word2vec_model.wv.vectors
    unknown_embedd = np.zeros(300)
    
    st.sidebar.subheader("Cluster Parameter")
    SUMMARY_SIZE = st.sidebar.slider("Berapa Jumlah Cluster?", 1, len(word_embedding(sentences)), 20)
    avg = []
    n = SUMMARY_SIZE
    vector = embedd_vectors[:SUMMARY_SIZE]
    st.subheader("Vector Word Embedding")
    st.dataframe(vector)
    n_clusters = len(sentences)//n
    modelmn = MiniBatchKMeans(n_clusters=n_clusters) #minibatch
    modelmn = modelmn.fit(vector)
    for j in range(n_clusters):
        idx = np.where(modelmn.labels_ == j)[0]
        avg.append(np.mean(idx))
    closest, _ = pairwise_distances_argmin_min(modelmn.cluster_centers_, vector)
    ordering = sorted(range(n_clusters), key=lambda k: avg[k])
    st.subheader("Closest & Ordering Cluster")
    col5, col6 = st.beta_columns([1, 1])
    col5.dataframe(closest)
    col6.dataframe(ordering)

    st.subheader("Summary Result")
#     summary = itemgetter(*ordering)(sentences)
#     hasilRingkasan = []
#     for sent in summary:
#         a = ' '.join(sent)
#         hasilRingkasan.append(a)
#     st.write(hasilRingkasan)
    summary = ' '.join([list_sentences[closest[idx]] for idx in ordering])
    st.write(summary)
    
elif genre == 'compareMethod':
    # Load word2vec pretrained
    st.sidebar.subheader("Word2vec Parameter")
    size_value = st.sidebar.slider("Berapa size?", 0, 200, len(sentences))
    mode_value = st.sidebar.selectbox("Pilih Mode", [1, 0])
    window_value = st.sidebar.slider("WIndows Size?", 0, 10, 3)
    iteration_value = st.sidebar.slider("iteration size?", 0, 100, 10) 
    word2vec_model = Word2Vec(sentences = sentences, size = size_value, sg = mode_value, window = window_value, min_count = 1, iter = iteration_value, workers = Pool()._processes)
    word2vec_model.init_sims(replace = True)
    embedd_vectors = word2vec_model.wv.vectors
    unknown_embedd = np.zeros(300)
    
    st.sidebar.subheader("Cluster Parameter")
    SUMMARY_SIZE = st.sidebar.slider("Berapa Jumlah Cluster?", 1, len(word_embedding(sentences)), 20)
    avg = []
    n = SUMMARY_SIZE
    vector = embedd_vectors[:SUMMARY_SIZE]
    n_clusters = len(sentences)//n
    modelmn = MiniBatchKMeans(n_clusters=n_clusters) #minibatch
    modelmn = modelmn.fit(vector)
    for j in range(n_clusters):
        idx = np.where(modelmn.labels_ == j)[0]
        avg.append(np.mean(idx))
    closest, _ = pairwise_distances_argmin_min(modelmn.cluster_centers_, vector)
    ordering = sorted(range(n_clusters), key=lambda k: avg[k])

    st.subheader("Summary Result Cluster")
    summary = ' '.join([list_sentences[closest[idx]] for idx in ordering])
    st.text(summary)
    
    # Sentence Ranking
    vector = [word_embedding(sentences[i]) for i in range(len(sentences))]
    vector_df = pd.DataFrame(vector)
    sentence_ranks = pagerank(vector_df)
    
    ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
    SUMMARY_SIZE = st.sidebar.slider("Berapa Jumlah Size?", 0, 10, 5)
    selected_sentences = sorted(ranked_sentence_indexes[:SUMMARY_SIZE])

    st.subheader("Summary Result Rank")
    ringkasan = itemgetter(*selected_sentences)(sentences)
    hasilSummary = [' '.join(sent) for sent in ringkasan]
    st.text(hasilSummary)
    
    from rouge import Rouge 
    hypothesis = (ringkasan)
    reference = (hasilSummary)
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    st.dataframe(scores)
