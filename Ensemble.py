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
###Importing Necessary Libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as cm
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

#Defining Classifiers
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=6)
clf3 = SVC(kernel='linear',probability = True, random_state = 0)
clf4 = MultinomialNB(alpha = 6.0)
#Ensemble Model
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3),
                                    ('mnb', clf4)], voting='soft', weights=[1, 1.5, 2,1.5])
#Creating Pipeline 
TFIDF_pipeline = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1,3))),
    ('tfidf', TfidfTransformer()),
    ('norm', Normalizer()),
    ('chi2', SelectKBest(chi2, k = 3000000)),
    ('clf', eclf)
])
#Fitting and Results    
TFIDF_pipeline.fit(X_train,np.ravel(y_train))
y_pred = TFIDF_pipeline.predict(X_val)
accuracy_score(y_val,y_pred)
cm = cm(y_val,y_pred)