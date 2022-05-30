# -*- coding: utf-8 -*-
"""
Research question: is textual data enough to differentiate between fake news & true news?

helpful source code for text classification & analysis:
https://github.com/Christophe-pere/Text-classification

## Imports & Data load
"""

# import needed libraries
import numpy as np
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer
from nltk import bigrams, trigrams
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter, defaultdict
from prettytable import PrettyTable
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.parsing.preprocessing import remove_stopwords,strip_punctuation
import spacy
from nltk.probability import FreqDist
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects
from sklearn.decomposition import LatentDirichletAllocation as LDA
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from nltk.metrics import ConfusionMatrix

# only needed when run in colab
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# loading the datasets
fake = pd.read_csv('Fake_news.csv')
true = pd.read_csv('True_news.csv')

corpus = fake.append(true)
corpus["fake"] = [1 for i in range(len(fake))]+[0 for i in range(len(true))]

# plot fake data
fake

# plot true data
true

print("Fake news info:\n")
print(fake.info(), "\n\n")
print("True news info:\n")
print(true.info())

fake.describe()

true.describe()

# remove duplicates
corpus = corpus.drop_duplicates(subset = ["text"])

fake = corpus[corpus["Fake"]==1]
print("Fake news length after duplicate removal:",len(fake))
true = corpus[corpus["Fake"]==0]
print("True news length after duplicate removal:",len(true))
# --> almost nothing filtered from true while fake news were reduced drastically.

# text length distribution per class
fake_lengths = [len(text) for text in fake["text"]]
true_lengths = [len(text) for text in true["text"]]
np.sort(true_lengths)

# plot text length distribution per class - bar chart
kwargs = dict(alpha=0.5, bins=100)

plt.hist(fake_lengths, **kwargs, color='r', label='Fake')
plt.hist(true_lengths, **kwargs, color='b', label='True')
plt.gca().set(title='Text length', ylabel='Frequency')
plt.legend();
#plt.savefig('Text_lentgh_distribution.png', bbox_inches='tight')

# plot text length distribution per class - boxplot
temp = pd.DataFrame({"length":fake_lengths+true_lengths,"type":["Fake" for i in range(len(fake_lengths))]+["True" for i in range(len(true_lengths))]})
sns.boxplot(temp["length"],temp["type"])

"""## Preprocessing"""

# remove stopwords
fake_destopped = [remove_stopwords(strip_punctuation(text)) for text in fake["text"]]
true_destopped = [remove_stopwords(strip_punctuation(text)) for text in true["text"]]

corpus_destopped = fake_destopped + true_destopped

# Sent Tokenization

# with stopwords
fake_sent_tokens_sw = [sent_tokenize(text) for text in fake["text"]]
true_sent_tokens_sw = [sent_tokenize(text) for text in true["text"]]

# without stopwords
fake_sent_tokens = [[remove_stopwords(strip_punctuation(sent)) for sent in text] for text in fake_sent_tokens_sw]
true_sent_tokens = [[remove_stopwords(strip_punctuation(sent)) for sent in text] for text in true_sent_tokens_sw]

# Word Tokenization 

# with stopwords
true_word_tokens_sw = [word_tokenize(text) for text in true["text"]]

# a lot of twitter posts in fake data --> use casual tokenizer to get rid of twitter handles and if existent two many repeating chars
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
fake_word_tokens_sw = [tknzr.tokenize(text) for text in fake["text"]]


# without stopwords
true_word_tokens = [word_tokenize(text) for text in true_destopped]

# a lot of twitter posts in fake data --> use casual tokenizer to get rid of twitter handles and if existent two many repeating chars
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
fake_word_tokens = [tknzr.tokenize(text) for text in fake_destopped]

# Lemmatization 

# with stopwords
lm = WordNetLemmatizer()
fake_word_tokens_lem_sw = [[lm.lemmatize(token) for token in text] for text in fake_word_tokens_sw]
true_word_tokens_lem_sw = [[lm.lemmatize(token) for token in text] for text in true_word_tokens_sw]

fake_lem_concat_sw = [" ".join(text) for text in fake_word_tokens_lem_sw]
true_lem_concat_sw = [" ".join(text) for text in true_word_tokens_lem_sw]

corpus_lem_sw = fake_lem_concat_sw + true_lem_concat_sw

# without stopwords
lm = WordNetLemmatizer()
fake_word_tokens_lem = [[lm.lemmatize(token) for token in text] for text in fake_word_tokens]
true_word_tokens_lem = [[lm.lemmatize(token) for token in text] for text in true_word_tokens]

fake_lem_concat = [" ".join(text) for text in fake_word_tokens_lem]
true_lem_concat = [" ".join(text) for text in true_word_tokens_lem]

corpus_lem = fake_lem_concat + true_lem_concat

# Downsample data

# with stopwords
fake_df_sw = pd.DataFrame({"text":fake_lem_concat_sw,"fake":fake["fake"]})
true_df_sw = pd.DataFrame({"text":true_lem_concat_sw,"fake":true["fake"]})

print("True dataset (with stopwords) size before Downsampling:",len(true_df_sw))
true_df_downsampled_sw = true_df_sw.sample(fake_df_sw.shape[0])
print("Fake dataset (with stopwords) size:",len(fake_df_sw))
print("True dataset (with stopwords) size after Downsampling:",len(true_df_downsampled_sw))
corpus_balanced_sw = pd.concat([true_df_downsampled_sw, fake_df_sw])
print("Total corpus (with stopwords) length after downsampling:",len(corpus_balanced_sw))

# without stopwords
fake_df = pd.DataFrame({"text":fake_lem_concat,"fake":fake["fake"]})
true_df = pd.DataFrame({"text":true_lem_concat,"fake":true["fake"]})

print("True dataset (with stopwords) size before Downsampling:",len(true_df))
true_df_downsampled = true_df.sample(fake_df.shape[0])
print("Fake dataset (with stopwords) size:",len(fake_df))
print("True dataset (with stopwords) size after Downsampling:",len(true_df_downsampled))
corpus_balanced = pd.concat([true_df_downsampled, fake_df])
print("Total corpus (with stopwords) length after downsampling:",len(corpus_balanced))

"""## Analysis

### Most frequent n-grams for each class
taken from https://github.com/Christophe-pere/Text-classification/blob/master/Text_Classification.ipynb

#### General Functions
"""

def get_top_n_words(corpus, n=None):
        '''
        Function to return a list of most frequent unigrams in documents
        @param corpus: (str or pandas.dataframe) documents 
        @param n: (int) number of most frequent unigrams
        @return: (list) most frequent unigrams
        '''
        vec = TfidfVectorizer().fit(corpus)             # bag of words
        bag_of_words = vec.transform(corpus)
        mean_words = bag_of_words.mean(axis=0) 
        words_freq = [(word, mean_words[0,idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:n]

def get_top_n_bigram(corpus, n=None):
        '''
        Function to return a list of most frequent bigrams in documents
        @param corpus: (str or pandas.dataframe) documents 
        @param n: (int) number of most frequent unigrams
        @return: (list) most frequent unigrams
        '''
        vec = TfidfVectorizer(ngram_range=(2, 2)).fit(corpus) 
        bag_of_words = vec.transform(corpus)
        mean_words = bag_of_words.mean(axis=0) 
        words_freq = [(word, mean_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:n]

def get_top_n_trigram(corpus, n=None):
        '''
        Function to return a list of most frequent trigrams in documents
        @param corpus: (str or pandas.dataframe) documents 
        @param n: (int) number of most frequent unigrams
        @return: (list) most frequent unigrams
        '''
        vec = TfidfVectorizer(ngram_range=(3, 3)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        mean_words = bag_of_words.mean(axis=0) 
        words_freq = [(word, mean_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:n]

def plot_top_word_freq(df,title):
    plt.figure(figsize=(10,6))
    plt.barh(df1["words"][::-1], df1["count"][::-1], color="tab:orange")
    plt.ylabel("Words")
    plt.xlabel("Average Frequency")
    plt.title(title)
    plt.grid(True,axis="x")
    #plt.savefig('Trigram_Avg_Freq_True.png', bbox_inches='tight')

"""#### Unigrams"""

# get Top 20 words frequencies for Fake news including stopwords 
common_words = get_top_n_words(fake["text"], 20)
df1 = pd.DataFrame(common_words, columns = ['words' , 'count'])

# plot Top 20 words frequencies
plot_top_word_freq(df1,"Top 20 words in Fake News before removing stopwords")

# get Top 20 words frequencies for True news including stopwords 
common_words = get_top_n_words(true["text"], 20)
df1 = pd.DataFrame(common_words, columns = ['words' , 'count'])

# plot Top 20 words frequencies
plot_top_word_freq(df1,"Top 20 words in True News before removing stopwords")

# get Top 20 words frequencies for Fake news without stopwords 
common_words_destopped = get_top_n_words(fake_destopped, 20)
df1 = pd.DataFrame(common_words_destopped, columns = ['words' , 'count'])

# plot Top 20 words frequencies
plot_top_word_freq(df1,"Top 20 words in Fake News after removing stopwords")

# get Top 20 words frequencies for Fake news without stopwords 
common_words_destopped = get_top_n_words(true_destopped, 20)
df1 = pd.DataFrame(common_words_destopped, columns = ['words' , 'count'])

# plot Top 20 words frequencies
plot_top_word_freq(df1,"Top 20 words in True News after removing stopwords")

"""#### Bigrams"""

# get Top 20 bigrams frequencies for Fake news including stopwords 
common_words = get_top_n_bigram(fake["text"], 20)
df1 = pd.DataFrame(common_words, columns = ['words' , 'count'])

# plot Top 20 bigram frequencies
plot_top_word_freq(df1,"Top 20 bigrams in Fake News before removing stopwords")

# get Top 20 bigrams frequencies for True news including stopwords 
common_words = get_top_n_bigram(true["text"], 20)
df1 = pd.DataFrame(common_words, columns = ['words' , 'count'])

# plot Top 20 bigram frequencies
plot_top_word_freq(df1,"Top 20 bigrams in True News before removing stopwords")

# get Top 20 bigrams frequencies for Fake news without stopwords 
common_words_destopped = get_top_n_bigram(fake_destopped, 20)
df1 = pd.DataFrame(common_words_destopped, columns = ['words' , 'count'])

# plot Top 20 bigram frequencies
plot_top_word_freq(df1,"Top 20 bigrams in Fake News after removing stopwords")

# get Top 20 bigrams frequencies for True news without stopwords 
common_words_destopped = get_top_n_bigram(true_destopped, 20)
df1 = pd.DataFrame(common_words_destopped, columns = ['words' , 'count'])

# plot Top 20 bigram frequencies
plot_top_word_freq(df1,"Top 20 bigrams in True News after removing stopwords")

"""#### Trigram (destopped only)"""

# get Top 20 trigrams frequencies for Fake news without stopwords
common_words_destopped = get_top_n_trigram(fake_destopped, 20)
df1 = pd.DataFrame(common_words_destopped, columns = ['words' , 'count'])

# plot Top 20 trigram frequencies
plot_top_word_freq(df1,"Top 20 trigrams in Fake News after removing stopwords")

# get Top 20 trigrams frequencies for True news without stopwords
common_words_destopped = get_top_n_trigram(true_destopped, 20)
df1 = pd.DataFrame(common_words_destopped, columns = ['words' , 'count'])

# plot Top 20 trigram frequencies
plot_top_word_freq(df1,"Top 20 trigrams in True News after removing stopwords")

"""### POS tagging

#### Fake News
"""

# Load spaCy English model
sp = spacy.load('en_core_web_sm')

# convert sentences into spaCy docs
#fake_spacy_docs = [[sp(sent) for sent in sentences] for sentences in tqdm(fake_sent_tokens_sw)]
#fake_spacy_docs[:1]
# spaCy docs are huge for some reason

# convert each spaCy doc into the respective POS tags
#fake_pos = [[str(s.pos_) for sent in sentences for s in sent] for sentences in tqdm(fake_spacy_docs)]

# save result in order not to run conversion it again
#np.save('fake_pos.npy', np.array(fake_pos, dtype=object))

# load tagged array
fake_pos = np.load('fake_pos.npy',allow_pickle=True)

# Join tags for each text
fake_pos_concat = [" ".join(tags) for tags in fake_pos]

# Calculate POS frequency
tfidf_pos = TfidfVectorizer()
fake_pos_vec = tfidf_pos.fit_transform(fake_pos_concat)

# Set up DataFrame with tags and average Frequencies
tags = tfidf_pos.get_feature_names()
pos_avg_frequencies = np.mean(fake_pos_vec.toarray(),axis=0)
df1 = pd.DataFrame({"POS":tags, "Frequency":pos_avg_frequencies})
df1 = df1.sort_values("Frequency",ascending=False)

# Plot average POS tag frequencies for Fake data
plt.figure(figsize=(10,6))
plt.barh(df1["POS"][::-1], df1["Frequency"][::-1])
plt.ylabel("POS",fontsize=14)
plt.xlabel("Average Frequency",fontsize=14)
plt.title("Average POS tag Frequency in Fake News",fontsize=18)
plt.grid(True, axis='x')
#plt.savefig('Avg_POS_Freq_Fake.png', bbox_inches='tight')

"""#### True News"""

# load tagged array
true_pos = np.load('true_pos.npy',allow_pickle=True)

# Join tags for each text
true_pos_concat = [" ".join(tags) for tags in true_pos]

# Calculate POS frequency
tfidf_pos = TfidfVectorizer()
true_pos_vec = tfidf_pos.fit_transform(true_pos_concat)

# Set up DataFrame with tags and average Frequencies
tags = tfidf_pos.get_feature_names()
pos_avg_frequencies = np.mean(true_pos_vec.toarray(),axis=0)
df1 = pd.DataFrame({"POS":tags, "Frequency":pos_avg_frequencies})
df1 = df1.sort_values("Frequency",ascending=False)

# Plot average POS tag frequencies for True data
plt.figure(figsize=(10,6))
plt.barh(df1["POS"][::-1], df1["Frequency"][::-1], color="tab:orange")
plt.ylabel("POS",fontsize=14)
plt.xlabel("Average Frequency",fontsize=14)
plt.title("Average POS tag Frequency in True News",fontsize=18)
plt.grid(True, axis="x")
#plt.savefig('Avg_POS_Freq_True.png', bbox_inches='tight')

"""### t-SNE
taken from https://www.scikit-yb.org/en/latest/api/text/tsne.html

#### General function
"""

# Utility function to visualize the outputs of t-SNE

def cluster_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(int)], alpha=0.6)
    plt.title("t-SNE clusters after stopword removal",fontsize=18)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        label = "Fake"
        if i==1:
            label = "True"
        txt = ax.text(xtext, ytext, label, fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=2, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    
    #plt.savefig('tsne_clusters.png', bbox_inches='tight')
    plt.show()
    return

"""#### With Stopwords"""

# vectorize corpus
corpus_vec_sw = TfidfVectorizer().fit_transform(corpus_lem_sw)

# first step dimensionality reduction with SVM -> 50 components
corpus_reduced_sw = TruncatedSVD(n_components=50).fit_transform(corpus_vec_sw)

corpus_reduced_sw.shape

# second step dimensionality reduction with tsne --> 2 components
corpus_reduced_sw = TSNE(n_components=2).fit_transform(corpus_reduced_sw)

#np.save('tsne_corpus_sw.npy', np.array(corpus_reduced_sw, dtype=object))

corpus_reduced_sw = np.load('tsne_corpus_sw.npy',allow_pickle=True)

corpus_reduced_sw.shape

# plot clusters to show difference between true and fake texts
cluster_scatter(corpus_reduced_sw, np.array(corpus["label"]))

"""#### Without Stopwords"""

# vectorize corpus
corpus_vec = TfidfVectorizer().fit_transform(corpus_lem)

# first step dimensionality reduction with SVM -> 50 components
corpus_reduced = TruncatedSVD(n_components=50).fit_transform(corpus_vec)

corpus_reduced.shape

# second step dimensionality reduction with tsne --> 2 components
corpus_reduced = TSNE(n_components=2).fit_transform(corpus_reduced)

#np.save('tsne_corpus.npy', np.array(corpus_reduced, dtype=object))

corpus_reduced = np.load('tsne_corpus.npy',allow_pickle=True)

corpus_reduced.shape

# plot clusters to show difference between true and fake texts
cluster_scatter(corpus_reduced, np.array(corpus["label"]))

"""### LDA"""

number_of_topics = 3

"""#### Overall"""

# vectorize corpus
tfidf_lda = TfidfVectorizer()
corpus_vec = tfidf_lda.fit_transform(corpus_lem)

# apply LDA
lda_aggregated = LDA(n_components=number_of_topics,random_state=42)
corpus_topic_vec = lda_aggregated.fit_transform(corpus_vec)

# plot Top 10 words for each generated topic
for index,topic in enumerate(lda_aggregated.components_):
    print(f'THE TOP 10 WORDS FOR TOPIC #{index}')
    print([tfidf_lda.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')

# obtain categories for all texts
corpus_categorized = corpus.copy()[["text","label"]]
corpus_categorized["topic"] = corpus_topic_vec.argmax(axis=1)

# get topic IDs and number of texts in the category
topics = corpus_categorized.groupby("topic")["text"].count().sort_values(ascending=True).index
counts = corpus_categorized.groupby("topic")["text"].count().sort_values(ascending=True)

# show value counts per topic
corpus_categorized.groupby(["topic","label"])["text"].count().sort_values(ascending=False)

# plot count per topic for each class
fake_count = [5,3678,17508]
true_count = [187,2876,14392]
index = topics
df = pd.DataFrame({'Fake': fake_count,
                   'True': true_count}, index=index)
ax = df.plot.barh(rot=0)
ax.grid(axis="x")
plt.ylabel("Topic")
plt.xlabel("Number of texts")
#plt.savefig('Topic_counts.png', bbox_inches='tight')

# show topic clusters with t-sne
palette = np.array(sns.color_palette("hls", number_of_topics))
f = plt.figure(figsize=(8, 8))
ax = plt.subplot(aspect='equal')
sc = ax.scatter(corpus_reduced[:,0], corpus_reduced[:,1], lw=0, s=40, c=palette[np.array(corpus_categorized["topic"]).astype(int)], alpha=0.5)
plt.xlim(-25, 25)
plt.ylim(-25, 25)
ax.axis('off')
ax.axis('tight')
#plt.savefig('Topic_cluster.png', bbox_inches='tight')

"""#### Fake News"""

# vectorize corpus
tfidf_lda = TfidfVectorizer()
corpus_vec = tfidf_lda.fit_transform(fake_lem_concat)

# apply LDA
lda_fake = LDA(n_components=number_of_topics,random_state=42)
corpus_topic_vec = lda_fake.fit_transform(corpus_vec)

# plot Top 10 words for each generated topic
for index,topic in enumerate(lda_fake.components_):
    print(f'THE TOP 10 WORDS FOR TOPIC #{index}')
    print([tfidf_lda.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')

# obtain categories for all texts
fake_categorized = corpus.copy()[["text","label"]][corpus["label"]==0]
fake_categorized["topic"] = corpus_topic_vec.argmax(axis=1)

# get topic IDs and number of texts in the category
topics = fake_categorized.groupby("topic")["text"].count().sort_values(ascending=True).index
counts = fake_categorized.groupby("topic")["text"].count().sort_values(ascending=True)

# show value counts per topic
fake_categorized.groupby("topic")["text"].count().sort_values(ascending=False)

# plot count per topic for Fake data
index = topics
df = pd.DataFrame({'Counts': counts}, index=index)
ax = df.plot.barh(rot=0)
ax.grid(axis="x")
plt.ylabel("Topic")
plt.xlabel("Number of texts")
#plt.savefig('Topic_counts.png', bbox_inches='tight')

"""could be done per category as well; also could use lemmatized text?

#### True News
"""

# vectorize corpus
tfidf_lda = TfidfVectorizer()
corpus_vec = tfidf_lda.fit_transform(true_lem_concat)

# apply LDA
lda_true = LDA(n_components=number_of_topics,random_state=42)
corpus_topic_vec = lda_true.fit_transform(corpus_vec)

# plot Top 10 words for each generated topic
for index,topic in enumerate(lda_true.components_):
    print(f'THE TOP 10 WORDS FOR TOPIC #{index}')
    print([tfidf_lda.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')

# obtain categories for all texts
true_categorized = corpus.copy()[["text","label"]][corpus["label"]==1]
true_categorized["topic"] = corpus_topic_vec.argmax(axis=1)

# get topic IDs and number of texts in the category
topics = true_categorized.groupby("topic")["text"].count().sort_values(ascending=True).index
counts = true_categorized.groupby("topic")["text"].count().sort_values(ascending=True)

# show value counts per topic
true_categorized.groupby("topic")["text"].count().sort_values(ascending=False)

# plot count per topic for True data
index = topics
df = pd.DataFrame({'Counts': counts}, index=index)
ax = df.plot.barh(rot=0)
ax.grid(axis="x")
plt.ylabel("Topic")
plt.xlabel("Number of texts")
#plt.savefig('Topic_counts.png', bbox_inches='tight')

'''palette = np.array(sns.color_palette("hls", number_of_topics))
f = plt.figure(figsize=(8, 8))
ax = plt.subplot(aspect='equal')
sc = ax.scatter(corpus_reduced[:,0], corpus_reduced[:,1], lw=0, s=40, c=palette[np.array(corpus_categorized["topic"]).astype(int)], alpha=0.5)
plt.xlim(-25, 25)
plt.ylim(-25, 25)
ax.axis('off')
ax.axis('tight')
#plt.savefig('Topic_cluster.png', bbox_inches='tight')'''

"""could be done per category as well; also could use lemmatized text?

### Sentiment analysis
"""

def get_sent_sentiment_scores(sent_tokens):
    '''
    function to get the average sentiment score for each text
    '''
    sentiment_scores = {}
    sia = SentimentIntensityAnalyzer()

    for count, sentences_list in enumerate(sent_tokens):
        if len(sentences_list) > 0:
            text_sentiment = {}
            n_sent = 0
            for indx, sent in enumerate(sentences_list):
                if len(sent)> 3:
                    sentiment_dict = sia.polarity_scores(sent)
                    text_sentiment[indx] = sentiment_dict
                n_sent +=1

            pos = 0
            neg = 0
            neu = 0
            if text_sentiment:
                for _, value in text_sentiment.items():
                    pos += value["pos"]
                    neg += value["neg"]
                    neu += value["neu"]
            sentiment_scores[count] = {"pos": (pos/n_sent)*100, "neg": (neg/n_sent)*100, "neu": (neu/n_sent)*100}
    return sentiment_scores

def get_overall_sentiment_score(sent_scores):
    '''
    function to get overall sentiment score of the dataset
    '''
    pos = 0
    neg = 0
    neu = 0
    for _, value in sent_scores.items():
        pos += value["pos"]
        neg += value["neg"]
        neu += value["neu"]

    return {"Positive": pos/len(sent_scores), "Negative": neg/len(sent_scores), "Neutral": neu/len(sent_scores)}

# obtain all sentiment scores
fake_sent_sentiment_scores = get_sent_sentiment_scores(fake_sent_tokens)
true_sent_sentiment_scores = get_sent_sentiment_scores(true_sent_tokens)

print(f"The overall sentiment in the fake dataset:\n", get_overall_sentiment_score(fake_sent_sentiment_scores))
print(f"The overall sentiment in the true dataset:\n", get_overall_sentiment_score(true_sent_sentiment_scores))

# plot overall sentiment scores for each class
index = get_overall_sentiment_score(fake_sent_sentiment_scores).keys()
df = pd.DataFrame({'Fake': get_overall_sentiment_score(fake_sent_sentiment_scores).values(),
                   'True': get_overall_sentiment_score(true_sent_sentiment_scores).values()}, index=index)
ax = df.plot.barh(rot=0,figsize=(10,5))
ax.grid(axis="x")
plt.ylabel("Sentiment",fontsize=12)
plt.xlabel("Score",fontsize=12)
plt.title("Sentiment scores for each class",fontsize=16)
#plt.savefig('Sentiment_scores.png', bbox_inches='tight')

"""## Classification"""

corpus_balanced = np.load('corpus_balanced.npy',allow_pickle=True)
corpus_balanced_sw = np.load('corpus_balanced_sw.npy',allow_pickle=True)

corpus_balanced = pd.DataFrame({"text":corpus_balanced[:,0],"fake":corpus_balanced[:,1]})
corpus_balanced_sw = pd.DataFrame({"text":corpus_balanced_sw[:,0],"fake":corpus_balanced_sw[:,1]})

"""#### General functions"""

def get_word_distribution(corpus):
    '''
    retrieve the word distribution for the given corpus
    '''
    all_words = []

    for message in corpus:
        words = word_tokenize(message)
        for w in words:
            all_words.append(w)
          
    all_words = nltk.FreqDist(all_words)

    # Print the overall number of words and the most common words
    print('Number of words: {}'.format(len(all_words)))
    print('Most common words: {}'.format(all_words.most_common(15)))
    return all_words

def find_features(text, word_features):
    '''
    Function to determine which of the word features are present in the given text
    '''
    words = word_tokenize(text)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

"""#### With stopwords"""

# get data and labels for Classification
X_sw = corpus_balanced_sw['text']
Y_sw = corpus_balanced_sw['fake']

X_sw[:2]

# get most common words from the corpus and use 1500 most common words as features
all_words_sw = get_word_distribution(X_sw)
word_features_sw = [x[0] for x in all_words_sw.most_common(1500)]

news_sw = list(zip(X_sw, Y_sw))

np.random.seed(1)
np.random.shuffle(news_sw)

# Call find_features function for each news text
feature_set_sw = [(find_features(text,word_features_sw), label) for (text, label) in tqdm(news_sw)]

# split data into train and test set
train_sw, test_sw = train_test_split(feature_set_sw, test_size=0.25, random_state=1)

# setup and run model
nb_sw = MultinomialNB()
nltk_model_sw = SklearnClassifier(nb_sw)
nltk_model_sw.train(train_sw)

# print accuracy of NB
accuracy_sw = nltk.classify.accuracy(nltk_model_sw, test_sw)
print("Naive Bayes model Accuracy:",accuracy_sw)

# Confusion Matrix
test_Y_sw = [text[1] for text in test_sw]
test_X_sw = [text[0] for text in test_sw]
test_predictions_sw = nltk_model_sw.classify_many(test_X_sw)
cm = ConfusionMatrix(test_Y_sw, test_predictions_sw)
print(cm)

"""#### Without stopwords"""

# get data and labels for Classification
X = corpus_balanced['text']
Y = corpus_balanced['fake']

# get most common words from the corpus and use 1500 most common words as features
all_words = get_word_distribution(X)
word_features = [x[0] for x in all_words.most_common(1500)]

news = list(zip(X, Y))

np.random.seed(1)
np.random.shuffle(news)

# Call find_features function for each news text
feature_set = [(find_features(text,word_features), label) for (text, label) in tqdm(news)]

# split data into train and test set
train, test = train_test_split(feature_set, test_size=0.25, random_state=1)

# setup and run model
nb = MultinomialNB()
nltk_model = SklearnClassifier(nb)
nltk_model.train(train)

# print accuracy of NB
accuracy = nltk.classify.accuracy(nltk_model, test)
print("Naive Bayes model Accuracy:",accuracy)

# Confusion Matrix
test_Y = [text[1] for text in test]
test_X = [text[0] for text in test]
test_predictions = nltk_model_sw.classify_many(test_X)
cm = ConfusionMatrix(test_Y, test_predictions)
print(cm)

"""### BERT

helpful source code for BERT implementation:
https://www.tensorflow.org/text/tutorials/classify_text_with_bert
"""

# install required dependencies
!pip install -q -U "tensorflow-text==2.8.*"
!pip install -q tf-models-official==2.7.0

# import required libraries

import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

# Separate into a training and testing dataset
train_true_df = true_df.sample(frac = 0.75)
test_true_df = true_df.drop(train_fake_df.index)
train_fake_df = fake.sample(frac = 0.75)
test_fake_df = fake.drop(train_true_df.index)

# Create required directory structure

d = test_fake_df['text']
file = 'set/test/fake/file{}.txt'

n = 0 # to number the files

for x in d:
    with open(file.format(n), 'w') as f:
        f.write(str(x))
    n += 1

d = test_true_df['text']
file = 'set/test/true/file{}.txt'

n = 0 # to number the files

for x in d:
    with open(file.format(n), 'w') as f:
        f.write(str(x))
    n += 1

d = train_true_df['text']
file = 'set/train/true/file{}.txt'

n = 0 # to number the files

for x in d:
    with open(file.format(n), 'w') as f:
        f.write(str(x))
    n += 1
d = train_fake_df['text']
file = 'set/train/fake/file{}.txt'

n = 0 # to number the files

for x in d:
    with open(file.format(n), 'w') as f:
        f.write(str(x))
    n += 1

# import BERT models for Pre-processing and Encoding
bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8'  #@param ["bert_en_uncased_L-12_H-768_A-12", "bert_en_cased_L-12_H-768_A-12", "bert_multi_cased_L-12_H-768_A-12", "small_bert/bert_en_uncased_L-2_H-128_A-2", "small_bert/bert_en_uncased_L-2_H-256_A-4", "small_bert/bert_en_uncased_L-2_H-512_A-8", "small_bert/bert_en_uncased_L-2_H-768_A-12", "small_bert/bert_en_uncased_L-4_H-128_A-2", "small_bert/bert_en_uncased_L-4_H-256_A-4", "small_bert/bert_en_uncased_L-4_H-512_A-8", "small_bert/bert_en_uncased_L-4_H-768_A-12", "small_bert/bert_en_uncased_L-6_H-128_A-2", "small_bert/bert_en_uncased_L-6_H-256_A-4", "small_bert/bert_en_uncased_L-6_H-512_A-8", "small_bert/bert_en_uncased_L-6_H-768_A-12", "small_bert/bert_en_uncased_L-8_H-128_A-2", "small_bert/bert_en_uncased_L-8_H-256_A-4", "small_bert/bert_en_uncased_L-8_H-512_A-8", "small_bert/bert_en_uncased_L-8_H-768_A-12", "small_bert/bert_en_uncased_L-10_H-128_A-2", "small_bert/bert_en_uncased_L-10_H-256_A-4", "small_bert/bert_en_uncased_L-10_H-512_A-8", "small_bert/bert_en_uncased_L-10_H-768_A-12", "small_bert/bert_en_uncased_L-12_H-128_A-2", "small_bert/bert_en_uncased_L-12_H-256_A-4", "small_bert/bert_en_uncased_L-12_H-512_A-8", "small_bert/bert_en_uncased_L-12_H-768_A-12", "albert_en_base", "electra_small", "electra_base", "experts_pubmed", "experts_wiki_books", "talking-heads_base"]

map_name_to_handle = {
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1'
}

map_model_to_preprocess = {
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
}

tfhub_handle_encoder = map_name_to_handle[bert_model_name]
tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

# define BERT models for Pre-processing and Encoding
bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
bert_model = hub.KerasLayer(tfhub_handle_encoder)

# Binary loss function configured for binary classification problem
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

# Define Model architecture
def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  return tf.keras.Model(text_input, net)

# Configuration of Adam optimizer and epochs 
epochs = 5
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

# Compile model
classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

# Model training
print(f'Training model with {tfhub_handle_encoder}')
history = classifier_model.fit(x=train_ds,
                               validation_data=val_ds,
                               epochs=epochs)

# Model evaluation
loss, accuracy = classifier_model.evaluate(test_ds)

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')