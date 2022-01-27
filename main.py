from gsdmm.mgp import MovieGroupProcess
import json
import numpy as np
from nltk.corpus import stopwords
import string
import re
import gensim

def compute_vocab_size(texts):
    vocab = set()
    for text in texts:
        for word in text:
            vocab.add(word)
    return len(vocab)

def create_docs(file_address):
    print("docs is being created...")
    with open(file_address, encoding="utf8") as f:
        tweet_lists = json.load(f)

    # parsing file into a list of tweets and discarding anything else.
    tweets = []
    for tweet_list in tweet_lists:
        if type(tweet_list) is list:
            tweets += tweet_list

    docs = []
    for tweet in tweets:
        if not tweet.get("text"):
            continue
        tweet_text = tweet["text"]

        # language check
        if tweet.get("lang") != "en":
            continue

        # remove URLs.
        words = tweet_text.split()
        non_url_words = []
        for word in words:
            if re.sub(r'http\S+', '', word) != '':
                non_url_words.append(word)
        tweet_text = " ".join(non_url_words)

        # replace punctuations with space
        all_punctuations = string.punctuation
        for punctuation in all_punctuations:
            if punctuation == "#":
                continue
            tweet_text = tweet_text.replace(punctuation, " ")

        # sentence/s --> list of words
        unique_words_in_tweet_text = list(set(tweet_text.split()))

        # lower case
        lowered_words = [word.lower() for word in unique_words_in_tweet_text]

        # stop word removal
        stops = set(stopwords.words('english'))
        filtered_words = [word for word in lowered_words if word not in stops]

        # remove #KyleRittenhouse
        if "#kylerittenhouse" in filtered_words:
            filtered_words.remove("#kylerittenhouse")

        # remove "amp" since it comes from '&amp;' (after punctuation removal)
        if "amp" in filtered_words:
            filtered_words.remove("amp")

        docs.append(filtered_words)
    
    print("... docs is successfully created.")
    return docs

# https://towardsdatascience.com/short-text-topic-modelling-lda-vs-gsdmm-20f1db742e14
# define function to get words in topics
def get_topics_lists(model, top_clusters, n_words):
    '''
    Gets lists of words in topics as a list of lists.
    
    model: gsdmm instance
    top_clusters:  numpy array containing indices of top_clusters
    n_words: top n number of words to include
    
    '''
    # create empty list to contain topics
    topics = []
    
    # iterate over top n clusters
    for cluster in top_clusters:
        #create sorted dictionary of word distributions
        sorted_dict = sorted(model.cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:n_words]
         
        #create empty list to contain words
        topic = []
        
        #iterate over top n words in topic
        for k,v in sorted_dict:
            #append words to topic list
            topic.append(k)
            
        #append topics to topics list    
        topics.append(topic)
    
    return topics

def main():
    # each doc in docs must be a unique list of tokens found in your short text document.
    docs = create_docs(file_address="kyle_tweets.json")
    vocab_size = compute_vocab_size(texts=docs)

    # create dictionary of all words in all documents
    dictionary = gensim.corpora.Dictionary(docs)

    # filter extreme cases out of dictionary (IDK the importance of this, let's keep it for now)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

    # create BOW dictionary
    bow_corpus = [dictionary.doc2bow(doc) for doc in docs]

    alpha_values = [0.0001, 0.01, 0.2, 0.4, 0.6, 0.8, 0.99]
    beta_values = [0.0001, 0.01, 0.2, 0.4, 0.6, 0.8, 0.99]
    number_of_clusters_values = [2, 3, 4, 5, 6, 7, 8, 9]

    # TO-DO: Modularize.
    for number_of_clusters in number_of_clusters_values:
        for alpha in alpha_values:
            for beta in beta_values:
                print("*******************")
                print("GSDMM results with K={}, alpha={} and beta={}".format(number_of_clusters, alpha, beta))
                mgp = MovieGroupProcess(K=number_of_clusters, alpha=alpha, beta=beta, n_iters=30)
                y = mgp.fit(docs, vocab_size)

                # print number of documents per topic
                doc_count = np.array(mgp.cluster_doc_count)
                print('Number of documents per topic :', doc_count)

                # Topics sorted by the number of document they are allocated to
                top_index = doc_count.argsort()[-15:][::-1]
                print('Most important clusters (by number of docs inside):', top_index)

                # define function to get top words per topic
                def top_words(cluster_word_distribution, top_cluster, values):
                    for cluster in top_cluster:
                        sort_dicts = sorted(cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:values]
                        print("\nCluster %s : %s"%(cluster, sort_dicts))

                # get top words in topics
                top_words(mgp.cluster_word_distribution, top_index, 20)

                # get topics to feed to coherence model
                topics = get_topics_lists(mgp, top_index, 20)

                # evaluate model using Topic Coherence score
                coherence_model = gensim.models.CoherenceModel(topics=topics, 
                                dictionary=dictionary, 
                                corpus=bow_corpus, 
                                texts=docs, 
                                coherence='c_v')

                coherence_value = coherence_model.get_coherence()  
                print("Coherence value is {} for K={}, alpha={} and beta={}.".format(coherence_value, number_of_clusters, alpha, beta))
                print("*******************")
                print()

if __name__ == "__main__":
    main()
