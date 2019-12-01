import pandas as pd
import nltk.tokenize as tkn
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

labels = [
    'alt.atheism',
    'comp.graphics',
    'comp.os.ms-windows.misc',
    # 'comp.sys.ibm.pc.hardware',
    # 'comp.sys.mac.hardware',
    # 'comp.windows.x',
    # 'misc.forsale',
    # 'rec.autos',
    # 'rec.motorcycles',
    # 'rec.sport.baseball',
    # 'rec.sport.hockey',
    # 'sci.crypt',
    # 'sci.electronics',
    # 'sci.med',
    # 'sci.space',
    # 'soc.religion.christian',
    # 'talk.politics.guns',
    # 'talk.politics.mideast',
    # 'talk.politics.misc',
    # 'talk.religion.misc'
]


def get_dataframe():
    df = pd.DataFrame()

    for label in labels:
        dir_path = 'data/20news-18828/' + label
        target = labels.index(label)
        for file in listdir(dir_path):
            file_path = join(dir_path, file)
            if isfile(file_path):
                with open(file_path, 'r', encoding='iso-8859-1') as f:
                    s = f.read()
                    tokens = tkn.WordPunctTokenizer().tokenize(s)
                    df = df.append(
                        pd.DataFrame(
                            {'text': s, 'label': label, 'target': target, 'tokens': [tokens], 'length': len(tokens)},
                            index=[file]))

    return df


def print_stats():
    df = get_dataframe()

    plt.figure()
    # df.plot.bar()
    df['length'].plot.hist()

    plt.show()
