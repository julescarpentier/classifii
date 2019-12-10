from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
from models.data_parsing import get_dataframe
from sklearn.neural_network import MLPClassifier

# affiche les mots les plus proches du mot desk
model = KeyedVectors.load_word2vec_format('data/wiki-news-300d-1M.vec/wiki-news-300d-1M.vec')
print(model.most_similar('desk'))

# parcours les mots de wiki-news et les mets dans un tableau
words = []
for word in model.vocab:
    words.append(word)

# affiche le vecteur correspondant au premier mot du tableau wiki-news
print("Vector components of a word: {}".format(
    model[words[0]]
))

# données que l'on veut véctoriser
df = get_dataframe()
sentences = df["tokens"]

# fonction permettant de vectoriser une phrase en fonction d'un model
def sent_vectorizer(sent, model):
    sent_vec = []
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw += 1
        except:
            pass

    return np.asarray(sent_vec) / numw

# vectorisation de nos tokens en fonction de notre model définit précédemment
V = []
for sentence in sentences:
    V.append(sent_vectorizer(sentence, model))

# sépare nos données vectorisées en une base d'entrainement et de test ( X données et Y cibles)

X_train = V[0:700] # 70pourcent
X_test = V[700:1000]  # 30pourcent
Y_train = [0]*700
Y_test = [0]*300

classifier = MLPClassifier(alpha=0.7, max_iter=400) # max_iter donne le nombre max d'iteration que l'on va faire
classifier.fit(X_train, Y_train) # return train MPL model

df_results = pd.DataFrame(data=np.zeros(shape=(1, 3)), columns=['classifier', 'train_score', 'test_score'])
train_score = classifier.score(X_train, Y_train)
test_score = classifier.score(X_test, Y_test)

print(classifier.predict_proba(X_test))
print(classifier.predict(X_test))

df_results.loc[1, 'classifier'] = "MLP"
df_results.loc[1, 'train_score'] = train_score
df_results.loc[1, 'test_score'] = test_score
print(df_results)
