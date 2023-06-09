import math
import re
from statistics import mean
import sys
import numpy as np
from nltk.stem import PorterStemmer
import json
ps = PorterStemmer()


def text_rank(sum_file, stop_file):
    threshold = 1.3
    total = 0

    sumfile = open(sum_file,"r+", encoding="utf8").readlines()
    for line in sumfile: 
        if total >= 50:
            break
        term_count = {}
        tf_table = {}
        tf_idf = {}
        doc_freq = {}
        idf_table = {}
        scores = {}

        stopfile = open(stop_file,"r+", encoding="utf8")
        stop = [w.replace('\n','').lower() for w in stopfile.readlines()]

        text = line
        text = text.replace('\n',' ')

        sentences=text.split(". ")
        N = len(sentences)

        sentences_clean=[re.sub(r'[^\w\s]','',sentence.lower()) for sentence in sentences]
        #sentence_tokens = [sentence.split(' ') for sentence in sentences_clean]
        sentence_tokens=[[words for words in sentence.split(' ') if (words not in stop)] for sentence in sentences_clean]

        if len(sentence_tokens) < 2:
            continue
        for sentence in sentence_tokens:
            sent_str = ' '.join([str(elem) for elem in sentence])
            term_count[sent_str] = {}
            seen = set()

            for word in sentence:
                word = ps.stem(word)
                term_count[sent_str][word] = term_count[sent_str].get(word,0) + 1
                if word not in seen:
                    doc_freq[word] = doc_freq.get(word,0) + 1
                    seen.add(word)

        #calculate document frequency
        for word in doc_freq.keys():
            idf_table[word] = {}
            idf_table[word] = math.log(N/doc_freq[word],10)

        #calculate term frequency
        for sent, count_table in term_count.items():
            tf_idf[sent] = {}
            total_count = len(count_table)
            for word, count in count_table.items():
                tf_idf[sent][word] = count / total_count * idf_table[word]

        for sent, count_table in tf_idf.items(): 
            if len(count_table) == 0:
                scores[sent] = 0
                continue
            scores[sent] = sum(count_table.values()) / len(count_table)
        
        mean_score = mean(scores.values())

        print_string = ""
        for sentence in sentences:
            sentence_clean = re.sub(r'[^\w\s]','',sentence.lower())
            sentence_tokens=[words for words in sentence_clean.split(' ') if (words not in stop)]
            sent_str = ' '.join([str(elem) for elem in sentence_tokens])
            if scores[sent_str] > (mean_score * threshold):
                print_string += " " + sentence + "."
        #tf_idf --> each sentence is key in dictionary, represented by vector of words w tf-idf 
        print(print_string)
        total +=1

def main():
    #sumfile, stopfile
    text_rank(sys.argv[1],sys.argv[2])

if __name__ == '__main__':
    main()
