import math
import re
from statistics import mean
import sys
import numpy as np
from nltk.stem import PorterStemmer
import json
ps = PorterStemmer()

def sent_extract(sum_file):
    total = 0

    sumfile = open(sum_file,"r+", encoding="utf8")
    for line in sumfile: 
        if total > 50:
            break
    
        data = json.loads(line)

        text = data['text']
        text = text.replace('\n',' ')

        print(text)
        total +=1

def main():
    #sumfile, stopfile
    sent_extract(sys.argv[1])

if __name__ == '__main__':
    main()
