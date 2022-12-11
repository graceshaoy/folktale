import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

import regex as re

import gensim
from gensim.models import Word2Vec
from gensim import corpora, models
from gensim.utils import effective_n_jobs

import string
import nltk
# nltk.download('stopwords')

import sklearn
from sklearn.manifold import TSNE

# from transformers import BertModel
# from transformers import AutoTokenizer

import time
import logging
import collections
from collections import Counter


#### DATA PROCESSING ####
STOP = set(nltk.corpus.stopwords.words('english') + list(string.punctuation) + ["\'","”","“","’","‘"])
def remove_stopwords(words):
    clean = []
    for w in words:
        if w not in STOP:
            clean.append(w)
    return clean

PUNC = [".",",","—"]
def clean_punc(string):
    for mark in PUNC:
        string = string.replace(mark," ").replace("  "," ")
    return string

# PHRASES = ["was a time","once upon a time"]
# def keep_phrases(string):
#     for p in PHRASES:
#         if string.lower().find(p) != -1:
#             string = string.replace(p,p.replace(" ",""))
#     return string

def get_bigrams(lemmas,min_count=10):
    bigram = models.Phrases(lemmas, min_count=min_count)
    bigram_phraser = models.phrases.Phraser(bigram)
    return set(bigram_phraser.phrasegrams.keys())

def make_bigrams(lemmas,min_count=10):
    bigram = models.Phrases(lemmas, min_count=min_count)
    bigram_mod = bigram.freeze()
    return [bigram_mod[doc] for doc in lemmas]

def to_wordlist(text):
    clean_strings = []
    for t in text:
#         t = keep_phrases(t)
        t = clean_punc(t)
        clean_strings.append(t)
    text = clean_strings
    text = [nltk.sent_tokenize(x.lower()) for x in text]
    text = sum(text,[])
    text = [nltk.word_tokenize(x) for x in text]
    clean = []
    for sent in text:
        clean.append(remove_stopwords(sent))
    return clean

def get_lemmas(text):
    lemmas = [nltk.stem.WordNetLemmatizer().lemmatize(t, get_wordnet_pos(t))
              for t in nltk.word_tokenize(text.lower()) if t not in STOP]
    return [l for l in lemmas if len(l) > 3]

def get_wordnet_pos(word):
    '''
    Tags each word with its Part-of-speech indicator -- specifically used for
    lemmatization in the get_lemmas function
    '''
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {'J': nltk.corpus.wordnet.ADJ,
                'N': nltk.corpus.wordnet.NOUN,
                'V': nltk.corpus.wordnet.VERB,
                'R': nltk.corpus.wordnet.ADV}

    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)

# def clean_countdf(df):
#     df.is_copy = False
#     only_letters = []
#     for i in df.index:
#         only_letters.append(re.sub(r"[^a-zA-Z]","",str(df['lemma'][i])))
#     df.loc[:,'lemma'] = only_letters

#     nonempty_index = []
#     for i in df.index:
#         if df['lemma'][i] != "":
#             nonempty_index.append(i)
#     df = df.loc[nonempty_index]
#     return df



def bigram_log_f(bgrm_df,msg):
    msg = msg.lower()
    for mark in [",",".",";",""]:
        msg = msg.replace(mark,"")
    bigrams = Counter(map(''.join,zip(msg,msg[1:])))
    msg_df = pd.DataFrame()
    msg_df['bigram'] = bigrams.keys()
    msg_df['count'] = bigrams.values()
    msg_df = msg_df.set_index('bigram')
    log_f_score = 0
    
    in_df = set(msg_df[msg_df.columns[0]])
    for b in msg_df.index:
        try:
            log_f_score += np.log(bgrm_df.loc[b][0])*msg_df.loc[b][0]
        except:
            log_f_score += -7
    return log_f_score

def make_posdict(lemma_counts,pos):
    pos_counts = {}
    for nation in lemma_counts:
        pos_counts[nation] = lemma_counts[nation].query('pos == @pos')
    for nation in pos_counts:
#         pos_counts[nation] = clean_countdf(pos_counts[nation]).reset_index(drop=True)
        pos_counts[nation]['portion'] = pos_counts[nation]['count'] / sum(pos_counts[nation]['count'])
        pos_counts[nation] = pos_counts[nation].drop([i for i in pos_counts[nation].index if pos_counts[nation]['count'][i] < 10])
    return pos_counts

def group_lemmacounts_bynation(pos_counts,nationlist):
    shared = pd.DataFrame(columns=['lemma'])
    for nation in pos_counts:
        shared = pd.merge(shared,pos_counts[nation][['lemma','portion']],on='lemma',how='outer',suffixes=("","_"+nation))
    shared.columns = list(shared.columns[:2]) + [col[8:] for col in shared.columns[2:]]
    shared = shared.reindex(columns = ['lemma'] + nationlist)
    shared['diff'] = [shared[nationlist].loc[0].max() - shared[nationlist].loc[0].min() for i in shared.index]
    shared = shared.drop_duplicates('lemma')
    shared = shared.fillna(0).set_index('lemma')

    return shared

def compare_ab(df,a,b):
    ab = df[['lemma',a,b]]
    ab['diff'] = ab[a] - ab[b]
    ab['dist'] = np.abs(ab['diff'])
    ab['avg'] = ab[[a,b]].mean(axis=1)
    ab_diff = ab.sort_values('diff',ascending=False).head(10)
    ba_diff = ab.sort_values('diff').head(10)
    ab_sim = ab.sort_values('avg',ascending=False).head(50).sort_values('dist').head(10)#.set_index('lemma')
    plt.figure(figsize=[15,5])
    width = 1/3
    
    ax1 = plt.subplot(1,3,1)
    plt.title(a.title() + " vs. " + b.title())
    xticks = np.arange(len(ab_diff))
    plt.bar(xticks,ab_diff[a],width=width)
    plt.bar(xticks+(1/3),ab_diff[b],width=width)
    plt.xticks(ticks=xticks, labels=ab_diff['lemma'],rotation=45)
    plt.ylabel('Frequency')
    plt.xlabel('Words used more in '+a.title()+" stories")
    
    ax2 = plt.subplot(1,3,2, sharey=ax1)
    xticks = np.arange(len(ba_diff))
    plt.bar(xticks,ba_diff[a],width=width)
    plt.bar(xticks+(1/3),ba_diff[b],width=width)
    plt.xticks(ticks=xticks, labels=ba_diff['lemma'],rotation=45)
    ax2.tick_params(axis='y',left=False,labelcolor='white')
    plt.xlabel('Words used more in '+b.title()+" stories")

def compare_liwc_hofstede(dimension,features,nation_liwc):
    liwc_cols = nation_liwc.columns[8:]
    nation_liwc_dim = pd.DataFrame(columns=['Culture',dimension,'Feature','score'])
    hofstede_dict = {'pdi':'Power Distance','idv':'Individualism','mas':'Masculinity','uai':'Uncertainty Avoidance','ltowvs':'Long Term Orientation','ivr':'Indulgence'}
    for col in liwc_cols:
        nation_liwc_dim = nation_liwc_dim.append(pd.DataFrame({'Culture':nation_liwc['Culture'],dimension:nation_liwc[dimension],'Feature':col,'score':nation_liwc[col]}))

    plot_df = pd.DataFrame()
    for col in features:
        plot_df = pd.concat([plot_df,nation_liwc_dim.query(f'Feature == "{col}"')])
    dots = alt.Chart(plot_df).mark_point(size=50).encode(
        x=alt.X(f'{dimension}:Q',scale=alt.Scale(zero=False),axis=alt.Axis(title=hofstede_dict[dimension])),
        y=alt.Y('score:Q',scale=alt.Scale(zero=False),axis=alt.Axis(title='Feature Score (normalized)')),
        color=alt.Color('Feature:N'),
        tooltip=['Culture','Feature']
    )
    line = alt.Chart(plot_df).mark_line().encode(
        x=alt.X(f'{dimension}:Q',scale=alt.Scale(zero=False),axis=alt.Axis(title=hofstede_dict[dimension])),
        y=alt.Y('mean(score)',scale=alt.Scale(zero=False),axis=alt.Axis(title='Feature Score (normalized)')),
        color = alt.Color('Feature:N')
    )
    dim_corr = pd.DataFrame(columns=['Feature','Correlation','P-Value'])
    for feature in liwc_cols:
        feature_df = nation_liwc_dim.query(f'Feature == "{feature}"')
        sr = spearmanr(feature_df[dimension],feature_df['score'])
        dim_corr.loc[len(dim_corr)] = [feature, sr[0],sr[1]]
    dim_corr = dim_corr.sort_values('Correlation',ascending=False)
    dim_corr = dim_corr[dim_corr['P-Value']<0.05]
    return dim_corr, dots+ line

    
def plot_word2vec_tsne(model, perplexity=40, n_iter=2500):
    '''
    Creates and TSNE model based on a Gensim word2vec model and plots it,
    given parameter inputs of perplexity and number of iterations.
    '''
    labels = []
    tokens = []

    for word in model.wv.key_to_index.keys():
        tokens.append(model.wv[word])
        labels.append(word)

    # Reduce 100 dimensional vectors down into 2-dimensional space
    tsne_model = TSNE(perplexity=perplexity, n_components=2, init='pca',
                      n_iter=n_iter, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.show()

def doc2vec_tsne(doc_model, labels, colorby='Nations', perplexity=40, n_iter=2500, n_components=2):
    tokens = []
    for i in range(len(doc_model.dv.vectors)):
        tokens.append(doc_model.dv.vectors[i])

    # Reduce 100 dimensional vectors down into 2-dimensional space
    tsne_model = TSNE(perplexity=perplexity, n_components=n_components, init='pca',
                      n_iter=n_iter, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    df = pd.DataFrame()
    for i in range(n_components):
        df['X'+str(i+1)] = [doc[i] for doc in new_values]


    # X = [doc[0] for doc in new_values]
    # y = [doc[1] for doc in new_values]
    # # Combine data into DataFrame, so that we plot it easily
    # df = pd.DataFrame({'X':X, 'y':y})
    # return new_values
    return df
    
def interactive_scatter(df, colorby='Nation'):
    scatter = alt.Chart(df).mark_point(filled=True, opacity=0.5).encode(
        alt.X('X0',scale=alt.Scale(zero=False)),
        alt.Y('X1',scale=alt.Scale(zero=False)),
        color=colorby,
        tooltip = [alt.Tooltip(colorby)]
    ).properties(
        width=60,
        height=60
    )
    return scatter

#### LDA ####
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=2):
    '''
    Computes Coherence values for LDA models with differing numbers of topics.

    Returns list of models along with their respective coherence values (pick
    models with the highest coherence)
    '''
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = models.ldamulticore.LdaMulticore(corpus=corpus,
                                                 id2word=dictionary,
                                                 num_topics=num_topics,
                                                 workers=effective_n_jobs(-1))
        model_list.append(model)
        coherence_model = models.coherencemodel.CoherenceModel(model=model,
                                                          corpus=corpus,
                                                          dictionary=dictionary,
                                                          coherence='u_mass')
        coherence_values.append(coherence_model.get_coherence())

    return model_list, coherence_values

def fill_topic_weights(df_row, bow_corpus, ldamodel):
    '''
    Fill DataFrame rows with topic weights for topics in songs.

    Modifies DataFrame rows *in place*.
    '''
    try:
        for i in ldamodel[bow_corpus[df_row.name]]:
            df_row[str(i[0])] = i[1]
    except:
        return df_row
    return df_row

def top_stories_by_topic(music_df, ldamodel, corpus, ntop=1):
    '''
    Finds the top "n" songs by topic, which we can use for
    understanding the types of songs included in a topic.
    '''
    topn_songs_by_topic = {}
    for i in range(len(ldamodel.print_topics())):
        # For each topic, collect the most representative song(s)
        # (i.e. highest probability containing words belonging to topic):
        top = sorted(zip(range(len(corpus)), ldamodel[corpus]),
                     reverse=True,
                     key=lambda x: abs(dict(x[1]).get(i, 0.0)))
        topn_songs_by_topic[i] = [j[0] for j in top[:ntop]]

        # Print out the topn songs for each topic and return their indices as a
        # dictionary for further analysis:
        print("Topic " + str(i))
        print(music_df[['nation','title','text','pdi','idv', 'mas','uai', 'ltowvs', 'ivr']].loc[topn_songs_by_topic[i]])
        print("*******************************")

    return topn_songs_by_topic

### LIWC ###
def readDict(dictionaryPath):
    '''
    Function to read in an LIWC-style dictionary
    '''
    catList = collections.OrderedDict()
    catLocation = []
    wordList = {}
    finalDict = collections.OrderedDict()

    # Check to make sure the dictionary is properly formatted
    with open(dictionaryPath, "r") as dictionaryFile:
        for idx, item in enumerate(dictionaryFile):
            if "%" in item:
                catLocation.append(idx)
        if len(catLocation) > 2:
            # There are apparently more than two category sections;
            # throw error and die
            sys.exit("Invalid dictionary format.")

    # Read dictionary as lines
    with open(dictionaryPath, "r") as dictionaryFile:
        lines = dictionaryFile.readlines()

    # Within the category section of the dictionary file, grab the numbers
    # associated with each category
    for line in lines[catLocation[0] + 1:catLocation[1]]:
        catList[re.split(r'\t+', line)[0]] = [re.split(r'\t+',
                                                       line.rstrip())[1]]

    # Now move on to the words
    for idx, line in enumerate(lines[catLocation[1] + 1:]):
        # Get each line (row), and split it by tabs (\t)
        workingRow = re.split('\t', line.rstrip())
        wordList[workingRow[0]] = list(workingRow[1:])

    # Merge the category list and the word list
    for key, values in wordList.items():
        if not key in finalDict:
            finalDict[key] = []
        for catnum in values:
            workingValue = catList[catnum][0]
            finalDict[key].append(workingValue)
    return (finalDict, catList.values())

def wordCount(data, dictOutput):
    '''
    Function to count and categorize words based on an LIWC dictionary
    '''
    finalDict, catList = dictOutput

    # Create a new dictionary for the output
    outList = collections.OrderedDict()

    # Number of non-dictionary words
    nonDict = 0

    # Convert to lowercase
    data = data.lower()

    # Tokenize and create a frequency distribution
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(data)

    fdist = nltk.FreqDist(tokens)
    wc = len(tokens)

    # Using the Porter stemmer for wildcards, create a stemmed version of data
    porter = nltk.PorterStemmer()
    stems = [porter.stem(word) for word in tokens]
    fdist_stem = nltk.FreqDist(stems)

    # Access categories and populate the output dictionary with keys
    for cat in catList:
        outList[cat[0]] = 0

    # Dictionaries are more useful
    fdist_dict = dict(fdist)
    fdist_stem_dict = dict(fdist_stem)

    # Number of classified words
    classified = 0

    for key in finalDict:
        if "*" in key and key[:-1] in fdist_stem_dict:
            classified = classified + fdist_stem_dict[key[:-1]]
            for cat in finalDict[key]:
                outList[cat] = outList[cat] + fdist_stem_dict[key[:-1]]
        elif key in fdist_dict:
            classified = classified + fdist_dict[key]
            for cat in finalDict[key]:
                outList[cat] = outList[cat] + fdist_dict[key]

    # Calculate the percentage of words classified
    if wc > 0:
        percClassified = (float(classified) / float(wc)) * 100
    else:
        percClassified = 0

    # Return the categories, the words used, the word count,
    # the number of words classified, and the percentage of words classified.
    return [outList, tokens, wc, classified, percClassified]

def liwc_features(text, liwc_dict, liwc_categories):
    '''
    Compute rel. percentage of LIWC 2007 categories:
    'affect', 'posemo', 'negemo', 'anx', 'anger', 'sad', 'social', 'family',
    'friend
    '''
    liwc_counts = wordCount(text, liwc_dict)

    return [liwc_counts[0][cat] / liwc_counts[2] for cat in liwc_categories].sum()