# -*- coding: utf-8 -*-
"""
Implementing N-gram, TF_IDF and VADER lexicon features
"""

##reading data

"""
1. change the directory to direcotry of positive and negative examples

"""

import os
import numpy as np

pos_train = []
pos_train_ID = []
neg_train = []
neg_train_ID = []

###should be the path of positive training examples
cwd = os.getcwd()
path = cwd+r"\train\neg"
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

###extracting word features uisng scikit module

from sklearn.model_selection import train_test_split   
X_train, X_val, y_train, y_val = train_test_split(training_x, training_y, train_size=0.8, test_size = 0.2)


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(ngram_range=(1,3)).fit(X_train)
x_train_counts = count_vect.transform(X_train)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(x_train_counts)
x_train_tfidf = tfidf_transformer.transform(x_train_counts)

from sklearn.preprocessing import Normalizer
normalizer_tranformer = Normalizer().fit(x_train_tfidf)
x_train_normalized = normalizer_tranformer.transform(x_train_tfidf)


######VADER features

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
import nltk

nltk.downloader.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

vader_scores = []

for sentence in X_train:
    #print(sentence)
    ss = sid.polarity_scores(sentence)
    vader_scores.append(ss)

###getting only positive and negative score
import pandas as pd
vader_feat = pd.DataFrame(vader_scores).values
vader_feat_select = vader_feat[:, [0, 2]]


vader_1D = np.subtract(vader_feat_select[:,1],vader_feat_select[:,0])
vader_1D = vader_1D.clip(min = 0)


#nlsen_feat_abs = nlsen_feat_select.clip(min=0)

from scipy import sparse
spars_vader_1D = sparse.csr_matrix(vader_1D)
spars_vader_1D = sparse.csr_matrix.transpose(spars_vader_1D)


from scipy.sparse import coo_matrix, hstack
all_feat = hstack([x_train_normalized,spars_vader_1D])

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
chi2_selector = SelectKBest(chi2, k=3000000)
X_kbest_train = chi2_selector.fit_transform(all_feat, y_train)

from sklearn.svm import SVC
clf = SVC(C = 1000, kernel = 'linear', random_state=0)
clf.fit(X_kbest_train,np.ravel(y_train)) 

#################################validation set

vadr_scores_val = []


for vsentence in X_val:
    
    ssv = sid.polarity_scores(vsentence)
    vadr_scores_val.append(ssv)


vaeder_feat_val = pd.DataFrame(vadr_scores_val).values
vaeder_feat_val = vaeder_feat_val[:, [0, 2]]

vaeder_feat_val_select = np.subtract(vaeder_feat_val[:,1],vaeder_feat_val[:,0])
vaeder_feat_val_select = vaeder_feat_val_select.clip(min = 0)

spars_vaeder_feat_val_select = sparse.csr_matrix(vaeder_feat_val_select)
spars_vaeder_feat_val_select = sparse.csr_matrix.transpose(vaeder_feat_val_select)

x_val_counts = count_vect.transform(X_val)
x_val_tfidf = tfidf_transformer.transform(x_val_counts)
x_val_normalized = normalizer_tranformer.transform(x_val_tfidf)

all_feat_val = hstack([x_val_normalized,spars_vaeder_feat_val_select])
X_kbest_val = chi2_selector.transform(all_feat_val)

val_predict_sck_svm = clf.predict(X_kbest_val)
np_y_val_predict_sck_svm = np.array(val_predict_sck_svm).reshape(len(val_predict_sck_svm),1)

from sklearn.metrics import accuracy_score
accuracy_score(y_val, np_y_val_predict_sck_svm)
