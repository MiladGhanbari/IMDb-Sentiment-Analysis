# -*- coding: utf-8 -*-
import os
import numpy as np
import time
import pickle
pos_train = []
pos_train_ID = []
neg_train = []
neg_train_ID = []

###should be the path of positive training examples
cwd = os.getcwd()
path = cwd + r"\train\pos"
os.chdir(path)

for filename in os.listdir(path):
    pos_train_ID.append(filename)
    file = open(filename, encoding="utf8")
    t = file.readlines()
    pos_train.append(str(t))
    
del path,filename,file,t

###should be the path of negative training data

path = cwd + r"\train\neg"
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
training_x = [x.lower() for x in training_x]
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(training_x, training_y,
                                                  train_size = 0.8,test_size = 0.2)
 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as cm


clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=6)
clf3 = SVC(kernel='linear',C=1000,random_state = 0)
clf4 = MultinomialNB(alpha = 6.0)
clf5 = LogisticRegression()

clfs = [clf1, clf2, clf3, clf4, clf5]

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
metricz = []
for clf in clfs:
    TFIDF_pipeline = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1,3))),
        ('tfidf', TfidfTransformer()),
        ('norm', Normalizer()),
        ('chi2', SelectKBest(chi2, k = 3000000)),
        ('clf', clf)
    ])
    t= time.time()
    scores = cross_val_score(TFIDF_pipeline,training_x,np.ravel(training_y),cv=5)
    TFIDF_pipeline.fit(X_train,np.ravel(y_train))
    y_pred = TFIDF_pipeline.predict(X_val)
    t1 = time.time()
    run = (t1-t)/60
    acc = accuracy_score(y_val,y_pred)
    metricz.append(["{}".format(clf),acc,cm(y_val,y_pred),scores])
    print("Classifier done with accuracy: {}".format(acc))