# -*- coding: utf-8 -*-
"""
Comparing Ngram features with TF*IDF uisng svm model
"""

##reading data

"""
change the directory to direcotry of positive and negative examples
"""

import os
import numpy as np

pos_train = []
pos_train_ID = []
neg_train = []
neg_train_ID = []

###should be the path of positive training examples
cwd = os.getcwd()
path = cwd+r"\train\pos"
os.chdir(path)

for filename in os.listdir(path):
    pos_train_ID.append(filename)
    file = open(filename, encoding="utf8")
    t = file.readlines()
    pos_train.append(str(t))
    
del path,filename,file,t

###should be the path of negative training data

path = cwd+r"\train\neg"
os.chdir(path)

for filename in os.listdir(path):
    neg_train_ID.append(filename)
    file = open(filename, encoding="utf8")
    t = file.readlines()
    neg_train.append(str(t))
    
del path,filename,file,t


###defining target labels
y_pos_train = np.ones((12500,1))
y_neg_train = np.zeros((12500,1))


#concatenate pos and neg training data to one array
training_x = np.concatenate((pos_train,neg_train), axis=0)
training_y = np.concatenate((y_pos_train,y_neg_train), axis = 0)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

###uisng only ngram features
n_gram_pipeline = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2), analyzer='word')),
    ('norm', Normalizer()),
    ('clf', SVC(kernel = 'linear', C = 1000, random_state=0)),
])
    

ngram_scores = cross_val_score(n_gram_pipeline, training_x, training_y, cv=4)

print("scores from ngram feature and linear svm:".format(ngram_scores.mean()) )


TFIDF_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('norm', Normalizer()),
    ('clf', SVC(kernel = 'linear', C = 1000, random_state=0)),
])
    

TFIDF_scores = cross_val_score(TFIDF_pipeline, training_x, training_y, cv=4)

print("scores from TFIDF feature and linear svm:".format(ngram_scores.mean()) )






