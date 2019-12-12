from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
from models.data_parsing import get_dataframe
from sklearn.neural_network import MLPClassifier

# affiche les mots les plus proches du mot desk
model = KeyedVectors.load_word2vec_format('data/wiki-news-300d-1M.vec/wiki-news-300d-1M.vec')
#print(model.most_similar('desk'))

# récupérer les mots seulement
mots = [x[0] for x in model]

# récupérer que les vecteurs correspondant aux mots
vect = [x[1:] for x in model]

# ajouter 1000 vecteurs de taille 0
vect = vect + np.zeros(300)*1000

# parcours les mots de wiki-news et les mots dans un tableau
# #words = []
# for word in model.vocab:
#     words.append(word)

# affiche le vecteur correspondant au premier mot du tableau wiki-news
#print("Vector components of a word: {}".format(
#    model[words[0]]
#))

# données que l'on veut véctoriser
#df = get_dataframe()
#sentences = df["tokens"]

# cibles de ce que l'on véctorize
#target = df["target"]

# fonction permettant de vectoriser tous les mots d'un document en fonction d'un model
#def sent_vectorizer(sent, model):
#    sent_vec = []
#    numw = 0
#    for w in sent:
#        try:
#            if numw == 0:
#                sent_vec = model[w]
#            else:
#                sent_vec = np.add(sent_vec, model[w])
#            numw += 1
#        except:
#            pass

#    return np.asarray(sent_vec) / numw

# vectorisation de nos tokens en fonction de notre model définit précédemment
#v = []
#for sentence in sentences:
#    v.append(sent_vectorizer(sentence, model))

# récupérer les mots seulement
#mots = [x[0] for x in v]

# récupérer que les vecteurs correspondant aux mots
#vect = [x[1:] for x in v]

# sépare nos données vectorisées en une base d'entrainement et de test ( x données et y cibles)

#x_train = v[0:700] # 70pourcent
#x_test = v[700:1000]  # 30pourcent
#y_train = target[0:700]
#y_test = target[700:1000]

#classifier = mlpclassifier(alpha=0.7, max_iter=400) # max_iter donne le nombre max d'iteration que l'on va faire
#classifier.fit(x_train, y_train) # return train mpl model

#df_results = pd.dataframe(data=np.zeros(shape=(1, 3)), columns=['classifier', 'train_score', 'test_score'])
#train_score = classifier.score(x_train, y_train)
#test_score = classifier.score(x_test, y_test)

#print(classifier.predict_proba(x_test))
#print(classifier.predict(x_test))

#df_results.loc[1, 'classifier'] = "mlp"
#df_results.loc[1, 'train_score'] = train_score
#df_results.loc[1, 'test_score'] = test_score
#print(df_results)
