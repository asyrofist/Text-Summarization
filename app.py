import streamlit as st
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import brown
from nltk.cluster.util import cosine_distance
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from operator import itemgetter
from function import TextCleaner
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from gensim.models import Word2Vec
from multiprocessing import Pool
from pywsd.cosine import cosine_similarity
from rouge import Rouge 

# nltk.download('brown')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stemming = PorterStemmer()
stops = set(stopwords.words("english"))
lem = WordNetLemmatizer()

st.set_page_config(
     page_title="Summarization",
)

# st.sidebar.subheader("Dataset parameter")
# banyak_data = st.sidebar.slider("Berapa Dataset", 0, len(brown.fileids()), 10)
# dataset = st.sidebar.selectbox("Choose Brown Dataset?", brown.fileids()[:banyak_data])
# sentences = brown.sents(dataset)
# list_sentences = [' '.join(sent) for sent in sentences]
# st.header("Summarization Corpus Brown")
# st.dataframe(list_sentences)

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

# cleaning text
def apply_cleaning_function_to_list(X):
    cleaned_X = []
    for element in X:
        cleaned_X.append(clean_text(element))
    return cleaned_X

def clean_text(raw_text):
    """This function works on a raw text string, and:
        1) changes to lower case
        2) tokenizes (breaks down into words
        3) removes punctuation and non-word text
        4) finds word stems
        5) removes stop words
        6) rejoins meaningful stem words"""
    
    # Convert to lower case
    text = raw_text.lower()
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Keep only words (removes punctuation + numbers)
    # use .isalnum to keep also numbers
    token_words = [w for w in tokens if w.isalpha()]
    
    # # Stemming
    # stemmed_words = [stemming.stem(w) for w in token_words]

    # Lemmatization
    lemma_words = [lem.lemmatize(w) for w in token_words]
    
    # Remove stop words
    meaningful_words = [w for w in lemma_words if not w in stops]

    # Rejoin meaningful stemmed words
    joined_words = ( " ".join(meaningful_words))
    
    # Return cleaned data
    return joined_words

st.subheader("Corpus Parameter")
text_dataset = st.text_area("Enter your Text", height=200, value = "Type Here", key="kalimatutama")
colutama, colkedua = st.beta_columns([2, 2])
sentences = nltk.sent_tokenize(text_dataset)
colutama.subheader("Dataset")
colutama.dataframe(sentences)
# Cleaning Text
text_to_clean = list(sentences)
cleaned_text = apply_cleaning_function_to_list(text_to_clean)
colkedua.subheader("cleaned")
colkedua.dataframe(cleaned_text)

st.sidebar.subheader("Method Parameter")
genre = st.sidebar.radio("What's your Method",('TextRank', 'disambiguationRank', 'disambiguationCluster', 'wordembedRank', 'wordembedCluster', 'validation'))
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
    SUMMARY_SIZE = st.sidebar.slider("Berapa Jumlah Size?", 0, len(ranked_sentence_indexes), 5)
    selected_sentences = sorted(ranked_sentence_indexes[:SUMMARY_SIZE])
    col4.dataframe(selected_sentences)

    st.subheader("Summary Result")
    summary = itemgetter(*selected_sentences)(sentences)
    st.write(' '.join(summary))
#     for sent in summary:
#         st.write(' '.join(sent))

elif genre == 'disambiguationRank':
    st.subheader("Disambiguation Ranking")
    
    col1, col2 = st.beta_columns([3, 1])
    disambiguation_df = []
    for angka in range(0, len(cleaned_text)):
        a = [cosine_similarity(cleaned_text[angka], cleaned_text[num]) for num in range(0, len(cleaned_text))]
        disambiguation_df.append(a)      

    hasil_disambiguation = pd.DataFrame(disambiguation_df)
    col1.write(hasil_disambiguation)
    sentence_ranks = pagerank(hasil_disambiguation)
    col2.write(sentence_ranks)
    
    # Load Word Sense Disambiguation 
    col3, col4 = st.beta_columns([3, 1])
    ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
    col3.subheader("Index Sentence")
    col3.dataframe(ranked_sentence_indexes)
    st.sidebar.subheader("Summary Parameter")
    SUMMARY_SIZE = st.sidebar.slider("Berapa Jumlah Size?", 0, len(ranked_sentence_indexes), 5)
    selected_sentences = sorted(ranked_sentence_indexes[:SUMMARY_SIZE])
    col4.subheader("Sentence Rank")
    col4.dataframe(selected_sentences)

    st.subheader("Summary Result")
    summary = itemgetter(*selected_sentences)(sentences)
#     st.write(summary)
    st.write(' '.join(summary))
#     for sent in summary:
#         st.write(' '.join(sent))        

elif genre == 'disambiguationCluster':
    # Load word2vec pretrained
    st.sidebar.subheader("Word2vec Parameter")
    disambiguation_df = []
    for angka in range(0, len(sentences)):
        a = [cosine_similarity(sentences[angka], sentences[num]) for num in range(0, len(sentences))]
        disambiguation_df.append(a)
          
    st.subheader("Disambiguation Parameter")
    hasil_disambiguation = pd.DataFrame(disambiguation_df)
    st.dataframe(hasil_disambiguation)
    vector = hasil_disambiguation
    SUMMARY_SIZE = st.sidebar.slider("Berapa Jumlah Cluster?", 1, len(sentences), len(sentences)//3)
    n = SUMMARY_SIZE
    avg = []
    n_clusters = len(sentences)//n
    modelkm = KMeans(n_clusters=n_clusters, init='k-means++')
    modelkm = modelkm.fit(vector)
    for j in range(n_clusters):
        idx = np.where(modelkm.labels_ == j)[0]
        avg.append(np.mean(idx))
    closest, _ = pairwise_distances_argmin_min(modelkm.cluster_centers_, vector)
    ordering = sorted(range(n_clusters), key=lambda k: avg[k])
    col5, col6 = st.beta_columns([1, 1])
    col5.subheader("Closest Cluster")
    col5.dataframe(closest)
    col6.subheader("Ordering Cluster")
    col6.dataframe(ordering)

    st.subheader("Summary Result")
#     summary = itemgetter(*ordering)(sentences)
#     hasilRingkasan = []
#     for sent in summary:
#         a = ' '.join(sent)
#         hasilRingkasan.append(a)
#     st.write(hasilRingkasan)
#     summary = ' '.join([list_sentences[closest[idx]] for idx in ordering])
    summary = ' '.join([sentences[closest[idx]] for idx in ordering])
    st.write(summary)
        
elif genre == 'wordembedRank':
    st.subheader("Sentence Ranking based on WordEmbedding")
    # Load Word Sense Disambiguation 
    st.sidebar.subheader("Word2vec Parameter")
    size_value = st.sidebar.slider("Berapa size?", 0, 200, len(sentences))
    mode_value = st.sidebar.selectbox("Pilih Mode", [1, 0])
    window_value = st.sidebar.slider("WIndows Size?", 0, 10, 3)
    iteration_value = st.sidebar.slider("iteration size?", 0, 100, 10) 
    word2vec_model = Word2Vec(sentences = cleaned_text, size = size_value, sg = mode_value, window = window_value, min_count = 1, iter = iteration_value, workers = Pool()._processes)
    word2vec_model.init_sims(replace = True)
    embedd_vectors = word2vec_model.wv.vectors
    
    col1, col2 = st.beta_columns([3, 1])
    vector = embedd_vectors[:size_value]
#     vector = embedd_vectors
#     vector = [word_embedding(sentences[i]) for i in range(len(sentences))]
    vector_df = pd.DataFrame(vector)
    col1.write(vector_df)
    sentence_ranks = pagerank(vector_df)
    col2.write(sentence_ranks)
    
    # Load Word Sense Disambiguation 
    col3, col4 = st.beta_columns([3, 1])
    ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
    col3.subheader("Index Sentence")
    col3.dataframe(ranked_sentence_indexes)
    st.sidebar.subheader("Summary Parameter")
    SUMMARY_SIZE = st.sidebar.slider("Berapa Jumlah Size?", 0, len(sentences), len(sentences)//3)
    selected_sentences = sorted(ranked_sentence_indexes[:SUMMARY_SIZE])
    col4.subheader("Sentence Rank")
    col4.dataframe(selected_sentences)

    st.subheader("Summary Result")
    summary = itemgetter(*selected_sentences)(sentences)
#     st.write(summary)
    st.write(' '.join(summary))
#     hasilSummary = [' '.join(sent) for sent in summary]
#     st.write(hasilSummary)
#     for sent in summary:
#         st.write(' '.join(sent))

elif genre == 'wordembedCluster':
    # Load word2vec pretrained
    st.sidebar.subheader("Word2vec Parameter")
    size_value = st.sidebar.slider("Berapa size?", 0, 200, len(cleaned_text))
    mode_value = st.sidebar.selectbox("Pilih Mode", [1, 0])
    window_value = st.sidebar.slider("WIndows Size?", 0, 10, 3)
    iteration_value = st.sidebar.slider("iteration size?", 0, 100, 10) 
    word2vec_model = Word2Vec(sentences = sentences, size = size_value, sg = mode_value, window = window_value, min_count = 1, iter = iteration_value, workers = Pool()._processes)
    word2vec_model.init_sims(replace = True)
    embedd_vectors = word2vec_model.wv.vectors
    unknown_embedd = np.zeros(300)
    
    st.sidebar.subheader("Cluster Parameter")
    SUMMARY_SIZE = st.sidebar.slider("Berapa Jumlah Cluster?", 1, len(word_embedding(sentences)), len(word_embedding(sentences))//3)
    avg = []
    n = SUMMARY_SIZE
    vector = embedd_vectors[:size_value]
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
    col5, col6 = st.beta_columns([1, 1])
    col5.subheader("Closest Cluster")
    col5.dataframe(closest)
    col6.subheader("Ordering Cluster")
    col6.dataframe(ordering)

    st.subheader("Summary Result")
#     summary = itemgetter(*ordering)(sentences)
#     hasilRingkasan = []
#     for sent in summary:
#         a = ' '.join(sent)
#         hasilRingkasan.append(a)
#     st.write(hasilRingkasan)
#     summary = ' '.join([list_sentences[closest[idx]] for idx in ordering])
    summary = ' '.join([sentences[closest[idx]] for idx in ordering])
    st.write(summary)
    
elif genre == 'validation':  
    st.subheader("Hypothesis")
    message1 = st.text_area("Enter your Text", height=200, value = text_dataset, key="kalimat1")
    st.subheader("Reference")
    message2 = st.text_area("Enter your Text", height=200, value = text_dataset, key="kalimat2")
    # penilaian rouge
    hypothesis = (message1)
    reference = (message2)
    rouge = Rouge()
    st.subheader("Nilai Rouge Validation")
    scores = rouge.get_scores(hypothesis, reference)
    st.dataframe(scores)
