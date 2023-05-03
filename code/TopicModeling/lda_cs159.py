#!/usr/bin/env python
# coding: utf-8
import numpy as np
import scipy
from scipy.stats import dirichlet, multinomial
import pandas as pd
from collections import Counter
import re
from nltk.corpus import stopwords

def preprocess(token: str):
    '''
    Function that formats and strips each word of junk characters and removes stopwords
    '''
    return re.sub(r'[^a-zA-Z\s-]','',token.lower().replace('\n', ' '))


print("---Loading data---")
df_data = pd.read_json('test.jsonl',lines=True)
#print(df_data.head())


print("--Fetching and filtering document tokens--")
raw_documents = df_data['text'].astype(str).apply(lambda x: preprocess(x)).to_numpy()
print(len(raw_documents))

stops = set(stopwords.words('english'))
print("--Removing stopwords--")
for r in range(len(raw_documents)):
        words = raw_documents[r].split(' ')
        words = [w for w in words if w not in stops]
        raw_documents[r] = ' '.join(words)
#print(raw_documents[:10])
print('--Creating vocab set--')
docs = [d.split() for d in raw_documents]
words = ' '.join(raw_documents).split(' ')
freq_dist = Counter(words)
#print(freq_dist.most_common()[-100:])
vocab = list(set(words))
print(f'{len(vocab)} words in initial vocab')
print('--Frequency Filtering--')
for r in range(len(raw_documents)):
        words = raw_documents[r].split(' ')
        words = [w for w in words if freq_dist[w] > 15 and w not in stops]
        raw_documents[r] = ' '.join(words)

docs = [d.split() for d in raw_documents]
#print(docs[0])
words = ' '.join(raw_documents).split(' ')
#print(words[0])
vocab = list(set(words))
vocab_mapping = {w:i for i,w in enumerate(vocab)}
print(f'Final vocab size: {len(vocab)}')
del words

#create word ids
print("--Vectorizing corpus--")
mapped_docs = []
longest_doc_length = 0
for doc in docs:
    new_doc = []
    vectorized_doc = doc
    doc_len = len(doc)       
    for i in range(doc_len):
        vectorized_doc[i] = vocab_mapping[doc[i]]
    longest_doc_length = max(longest_doc_length, doc_len)
    mapped_docs.append(vectorized_doc)

print(f'{len(mapped_docs)} documents have been vocab mapped')


#Number of topics
K = 50
#Number of iterations through the entire corpus
num_iterations = 50
#topic-word matrix
tw_matrix = np.zeros((K,len(vocab)), dtype=np.int16)
#topic assignment history
assignments = np.zeros((len(mapped_docs), longest_doc_length, num_iterations+1 ), dtype=np.int16)
#document-topic matrix
dt_matrix = np.zeros((len(docs),K), dtype=np.int16)

#Randomly intitialize matrices
for d in range(len(docs)):
    for w in range(len(mapped_docs[d])):
        ti = np.random.randint(0,K)
        assignments[d,w,0] = ti
        wi = int(mapped_docs[d][w])
        tw_matrix[ti, wi] += 1
        dt_matrix[d,ti] += 1
    
#Model paramters
alpha = 0.5
eta = 1

#calculating P(z_i|*)
print('-----Running Gibbs Sampling Loop-----')
for iteration in range(num_iterations):
    print(f'Iteration {iteration}/{num_iterations}')
    for d_i in range(len(mapped_docs)):
        for w_i in range(len(mapped_docs[d_i])):
            init_topic = int(assignments[d_i, w_i, iteration])
            word_id = mapped_docs[d_i][w_i] 
            #z_-i term
            dt_matrix[d_i, init_topic] -= 1
            tw_matrix[init_topic, word_id] -= 1
            #word topic means
            wt_means = (tw_matrix[:, word_id] + eta) / (tw_matrix.sum(axis=1) + len(vocab)*eta)
            dt_means = (dt_matrix[d_i,:]+alpha) / (dt_matrix[d_i,:].sum() + K*alpha )
            probs = wt_means*dt_means
            #Normalize, necessary due to rounding errors
            probs = probs/probs.sum()
            #Multinomial draws
            new_topic = np.argmax(np.random.multinomial(1,probs))
            dt_matrix[d_i, new_topic] += 1
            tw_matrix[new_topic, word_id] += 1
            #update topic assignment list
            assignments[d_i,w_i, iteration+1] = new_topic
print("finished")

print('Topics')
word_lists = []
for k in range(K):
    topic_k_words = tw_matrix[k, :]
    #Get top 10 words for topic k
    top_words_ind = np.argpartition(topic_k_words, -50)[-50:]
    top_words = [vocab[v] for v in top_words_ind]
    word_lists.append(top_words)
    print(f"Topic {k}: {', '.join(top_words)}")

np.savez('topic_model_artifacts.npz',vocab=np.array(vocab),topic_word=tw_matrix,doc_topic=dt_matrix)


