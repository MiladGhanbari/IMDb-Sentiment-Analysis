# -*- coding: utf-8 -*-
"""
Implementing N-gram, TF_IDF and Bing Liu's lexicon features
"""


"""
change the directory to access the lexicon collections. 
positive sentiment : pos_sent.txt
negative sentiment: neg_sent.txt

"""

##reading data

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


from sklearn.model_selection import train_test_split   
X_train, X_val, y_train, y_val = train_test_split(training_x, training_y, train_size=0.8, test_size = 0.2)

path = cwd+r"\opinion-lexicon-English"
os.chdir(path)
file = open('sent_neg.txt', 'r')
neg_sent_org = file.read().splitlines()
del file
file = open('sent_pos.txt', 'r')
pos_sent = file.read().splitlines()

##randomly picking some instances from the negative collection

import random
def random_subset( iterator, K ):
    result = []
    N = 0

    for item in iterator:
        N += 1
        if len( result ) < K:
            result.append( item )
        else:
            s = int(random.random() * N)
            if s < K:
                result[ s ] = item

    return result


neg_sent = random_subset(neg_sent_org,len(pos_sent))

###extracting the lexicon feature for training set
A_x_train = np.ndarray.tolist(X_train)
A_x_valid = np.ndarray.tolist(X_val)

print("done donwloading")

#creat feature vectors

pos_sent_feat = np.zeros((len(A_x_train),len(pos_sent))) 
neg_sent_feat = np.zeros((len(A_x_train),len(neg_sent))) 


for comments in A_x_train:
        for pos_words in pos_sent:
            if pos_words in comments:
                pos_sent_feat[A_x_train.index(comments),pos_sent.index(pos_words)] = comments.count(pos_words)
            
del comments
del pos_words

for comments in A_x_train:
        for neg_words in neg_sent:
            if neg_words in comments:
                neg_sent_feat[A_x_train.index(comments),neg_sent.index(neg_words)] = comments.count(neg_words)
                  
del comments
del neg_words


print("done with senti")

x_feat = np.concatenate((pos_sent_feat,neg_sent_feat), axis = 1)

###extracting the lexicon feature for validation set

pos_sent_feat_val = np.zeros((len(A_x_valid),len(pos_sent))) 
neg_sent_feat_val = np.zeros((len(A_x_valid),len(neg_sent))) 


for comments in A_x_valid:
        for pos_words in pos_sent:
            if pos_words in comments:
                pos_sent_feat_val[A_x_valid.index(comments),pos_sent.index(pos_words)] = comments.count(pos_words)
            
del comments
del pos_words


for comments in A_x_valid:
    for neg_words in neg_sent:
        if neg_words in comments:
            neg_sent_feat_val[A_x_valid.index(comments),neg_sent.index(neg_words)] = comments.count(neg_words)
                  
del comments
del neg_words


x_feat_val = np.concatenate((pos_sent_feat_val,neg_sent_feat_val), axis = 1)

###############extracting features using scikit learn

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(ngram_range=(1,3)).fit(X_train)
x_train_counts = count_vect.transform(X_train)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(x_train_counts)
x_train_tfidf = tfidf_transformer.transform(x_train_counts)

from sklearn.preprocessing import Normalizer
normalizer_tranformer = Normalizer().fit(x_train_tfidf)
x_train_normalized = normalizer_tranformer.transform(x_train_tfidf)

#########concatenating lexicon features with other text features 

from scipy import sparse
spar_x_feat = sparse.csr_matrix(x_feat)

from scipy.sparse import coo_matrix, hstack
all_feat = hstack([x_train_normalized,spar_x_feat])

########choosing top 3M fratures

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
chi2_selector = SelectKBest(chi2, k=3000000)
X_kbest_train = chi2_selector.fit_transform(all_feat, y_train)

#######training linear SVM
from sklearn.svm import SVC
clf = SVC(C = 1000, kernel = 'linear', random_state=0)
clf.fit(X_kbest_train,np.ravel(y_train)) 

###############################################

##getting the accuracy on the validation set
x_val_counts = count_vect.transform(X_val)
x_val_tfidf = tfidf_transformer.transform(x_val_counts)
x_val_normalized = normalizer_tranformer.transform(x_val_tfidf)

spar_x_feat_val = sparse.csr_matrix(x_feat_val)
all_feat_val = hstack([x_val_normalized,spar_x_feat_val])
X_kbest_val = chi2_selector.transform(all_feat_val)

val_predict_sck_svm = clf.predict(X_kbest_val)
np_y_val_predict_sck_svm = np.array(val_predict_sck_svm).reshape(len(val_predict_sck_svm),1)

from sklearn.metrics import accuracy_score
accuracy_score(y_val, np_y_val_predict_sck_svm)




