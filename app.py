import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import brown
from nltk.cluster.util import cosine_distance
from operator import itemgetter
from function import TextCleaner
from pywsd.cosine import cosine_similarity

nltk.download('brown')

st.header("Summarization Corpus Brown")
sentences = brown.sents('ca01')
list_sentences = [' '.join(sent) for sent in sentences]
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
  
st.subheader("Sentence Ranking")
col1, col2 = st.beta_columns([3, 1])
S = build_similarity_matrix(sentences)
col1.write(S)
sentence_ranks = pagerank(S)
col2.write(sentence_ranks)

st.subheader("disamiguation")
# Load Word Sense Disambiguation 
from pywsd.cosine import cosine_similarity
disambiguation_df = []
for angka in range(0, len(list_sentences)):
    a = [cosine_similarity(list_sentences[angka], list_sentences[num]) for num in range(0, len(list_sentences))]
    disambiguation_df.append(a)      

hasil_disambiguation = pd.DataFrame(disambiguation_df)
st.write(hasil_disambiguation)

## Load Word Sense Disambiguation 
# st.subheader("Index Sentence Ranking")
# col3, col4 = st.beta_columns([3, 1])
# ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
# col3.dataframe(ranked_sentence_indexes)
# SUMMARY_SIZE = st.slider("Berapa Jumlah Size?", 0, 10, 5)
# # SUMMARY_SIZE = 5
# selected_sentences = sorted(ranked_sentence_indexes[:SUMMARY_SIZE])
# col4.dataframe(selected_sentences)

# st.subheader("Summary Result")
# summary = itemgetter(*selected_sentences)(sentences)
# for sent in summary:
#     st.write(' '.join(sent))
