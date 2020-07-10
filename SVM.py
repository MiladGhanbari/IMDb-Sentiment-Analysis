# -*- coding: utf-8 -*-


# clean code 0.907
import os
import pickle
import numpy as np

"""
 proposed model SVM, Liear, C=1000
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

##split the data   
from sklearn.model_selection import train_test_split   
X_train, X_val, y_train, y_val = train_test_split(training_x, training_y, train_size=0.8, test_size = 0.2)


###test values:
####3.5M,4M features + C 5000 start with
##########################train

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(ngram_range=(1,3)).fit(X_train)
x_train_counts = count_vect.transform(X_train)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(x_train_counts)
x_train_tfidf = tfidf_transformer.transform(x_train_counts)

from sklearn.preprocessing import Normalizer
normalizer_tranformer = Normalizer().fit(x_train_tfidf)
x_train_normalized = normalizer_tranformer.transform(x_train_tfidf)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
chi2_selector = SelectKBest(chi2, k=3000000)
X_kbest_train = chi2_selector.fit_transform(x_train_normalized, y_train)

from sklearn.svm import SVC
clf = SVC(C = 1000, kernel = 'linear',random_state=0)
clf.fit(X_kbest_train,np.ravel(y_train)) 

########################valid

x_val_counts = count_vect.transform(X_val)
x_val_tfidf = tfidf_transformer.transform(x_val_counts)
x_val_normalized = normalizer_tranformer.transform(x_val_tfidf)
X_kbest_val = chi2_selector.transform(x_val_normalized)

val_predict_sck_svm = clf.predict(X_kbest_val)
np_y_val_predict_sck_svm = np.array(val_predict_sck_svm).reshape(len(val_predict_sck_svm),1)

from sklearn.metrics import accuracy_score
accuracy_score(y_val, np_y_val_predict_sck_svm)


#######################final train before submission

count_vect_final = CountVectorizer(ngram_range=(1,3)).fit(training_x)
x_train_counts_final = count_vect_final.transform(training_x)

tfidf_transformer_final = TfidfTransformer().fit(x_train_counts_final)
x_train_tfidf_final = tfidf_transformer_final.transform(x_train_counts_final)
normalizer_tranformer_final = Normalizer().fit(x_train_tfidf_final)
x_train_normalized_final = normalizer_tranformer_final.transform(x_train_tfidf_final)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
chi2_selector_final = SelectKBest(chi2, k=3000000)
X_kbest_train_final = chi2_selector_final.fit_transform(x_train_normalized_final, training_y)

from sklearn.svm import SVC
clf_final = SVC(C = 1000, kernel = 'linear', random_state=0)
clf_final.fit(X_kbest_train_final,np.ravel(training_y))

########################getting the IDs for submission

"""
change the direction to access test folder 
"""
path = cwd+r"\test"
os.chdir(path)

test = []
test_ID = []

for filename in os.listdir(path):
    test_ID.append(filename)
    file = open(filename, encoding="latin-1")
    t = file.readlines()
    test.append(str(t))

test_org = test
test = test_org[0:25000]

x_test_counts = count_vect_final.transform(test)
x_test_tfidf = tfidf_transformer_final.transform(x_test_counts)
x_test_normalized = normalizer_tranformer_final.transform(x_test_tfidf)
X_kbest_test = chi2_selector_final.transform(x_test_normalized)

test_predict_sck_svm = clf_final.predict(X_kbest_test)
np_test_predict_sck_svm = np.array(test_predict_sck_svm).reshape(len(test_predict_sck_svm),1)


np.savetxt(os.getcwd()+r"\largeCc.out",np_test_predict_sck_svm, newline = '\n' )

