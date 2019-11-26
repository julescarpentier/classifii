import pandas as pd
import nltk.tokenize as nltk
import nltk.tokenize.WordPunctTokenizer as wpt
from os import listdir
from os.path import isfile, join

labels = [
    'alt.atheism',
    'comp.graphics',
    'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'comp.windows.x',
    'misc.forsale',
    'rec.autos',
    'rec.motorcycles',
    'rec.sport.baseball',
    'rec.sport.hockey',
    'sci.crypt',
    'sci.electronics',
    'sci.med',
    'sci.space',
    'soc.religion.christian',
    'talk.politics.guns',
    'talk.politics.mideast',
    'talk.politics.misc',
    'talk.religion.misc'
]

df = pd.DataFrames()

for label in labels:
    dir = '../20news-18828/' + label
    for file in listdir(dir):
        file_path = join(dir, file)
        if isfile(file_path):
            with open(file_path, 'r', encoding='iso-8859-1') as f:
                s = f.read()
                # tokens = wpt.tokenize(s)
                print(s)
