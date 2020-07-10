# -*- coding: utf-8 -*-


import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from BernoulliNaiveClassifier import BernoulliNaiveBayse
from sklearn.feature_extraction.text import TfidfVectorizer

################################### Paths of data ###################################################
cwd = os.getcwd()
data_path_training_neg =cwd + r'\train\neg'
data_path_training_pos =cwd + r'\train\pos'
data_path_test =cwd + r'C\test'
File_names_neg =[];
File_names_pos =[];
File_names_test = []

# training and test variables definition
Training_data_text=[]
Training_data_class=[]
Test_data_text=[]
Neg_sent_words = []
Pos_sent_words = []

############################### Loading Data ################################################## 

# reading and sorting neg training data from txt files
for root, dirs, files in os.walk(data_path_training_neg):
    for name in files:
        if name.endswith((".txt")):
            File_names_neg.append(name)
File_names_sorted_neg = sorted(File_names_neg,key = lambda x: int(x.split('_')[0]))
for i in range(len(File_names_sorted_neg)):
    txtfile_path = data_path_training_neg + '\\' + File_names_sorted_neg[i]
    d = open(txtfile_path, 'r', encoding="utf8")
    Training_data_text.append(d.read())
    Training_data_class.append(0)
    d.close()
 
# reading and sorting pos training data from txt files
for root, dirs, files in os.walk(data_path_training_pos):
    for name in files:
        if name.endswith((".txt")):
            File_names_pos.append(name)
File_names_sorted_pos = sorted(File_names_pos,key = lambda x: int(x.split('_')[0]))
for i in range(len(File_names_sorted_pos)):
    txtfile_path = data_path_training_pos + '\\' + File_names_sorted_pos[i]
    d = open(txtfile_path, 'r', encoding="utf8")
    Training_data_text.append(d.read())
    Training_data_class.append(1)
    d.close()
    
# reading and sorting test data from txt files
for root, dirs, files in os.walk(data_path_test):
    for name in files:
        if name.endswith((".txt")):
            File_names_test.append(name)
File_names_sorted = sorted(File_names_test)
for i in range(len(File_names_sorted)):
    txtfile_path = data_path_test + '\\' + File_names_sorted[i]
    d = open(txtfile_path, 'r', encoding="utf8")
    Test_data_text.append(d.read())
    d.close()
	
    
########################## Spliting Data into Train and Validation #############################
X_train, X_val, Y_train, Y_val = train_test_split(Training_data_text, Training_data_class, train_size = 0.8, test_size = 0.2)


########################### Preprocessing and Creating Feature Matrix ###########################
ngram_vectorizer  = TfidfVectorizer(binary=True, ngram_range=(1, 3), max_features=10000)
XTrain = ngram_vectorizer.fit_transform(X_train).toarray()
XVal = ngram_vectorizer.transform(X_val).toarray()


########################## Bernoulli Naive Bayse Classifier #####################################
Classifier = BernoulliNaiveBayse()
X_train = np.asarray(XTrain)
Y_train = np.asarray(Y_train)
X_val = np.asarray(XVal)
y_pred = Classifier.BNB_Classifier(XTrain, Y_train, X_val) 
accuracy = accuracy_score(Y_val, y_pred)
print('BNB accuracy: {}'.format(accuracy))