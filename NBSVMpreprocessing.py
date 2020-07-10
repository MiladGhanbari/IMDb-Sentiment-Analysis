from nltk.corpus import stopwords
import numpy as np
import os
from sklearn.model_selection import train_test_split


def create_bow(sentence, vocab_list, gram):
    word_list = tokenize(sentence, gram)
    bow = np.zeros(len(vocab_list))
    
    for word in word_list:
         if word in vocab_list:
            bow[vocab_list[word]] = 1
    return bow

def rm_stopwords(word_list):
    return [word for word in word_list if word not in stopwords.words('english')]

def tokenize(sent, grams):
    words_list = rm_stopwords(sent.split())
    sent_tok = []
    for gram in range(1, grams + 1):
        for i in range(len(words_list) + 1 - gram):
            sent_tok.append("-".join(words_list[i:i + gram]))
    return sent_tok
    
"""
Loads the raw data 
"""
def build_vocab(gram):
    
    #Load data
    cwd = os.getcwd()
    # the paths of data
    data_path_training_neg = cwd + r'\train\neg'
    data_path_training_pos = cwd + r'\train\pos'
    data_path_test = cwd + r'\test'
    File_names_neg =[];
    File_names_pos =[];
    File_names_test = []
    
    # training and test variable definition
    Training_data_text=[]
    Training_data_class=[]
    Test_data_text=[]
    
    # reading neg training data from txt files
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
        
    # reading pos training data from txt files
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
        
    
    # reading test data from txt files
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
    
    # spliting train and valid data
    X_train, X_val, Y_train, Y_val = train_test_split(Training_data_text, Training_data_class, train_size = 0.8, test_size = 0.2)
    
    # creating training data dictionary
    Training_data_dic = {}
    for i in range(len(X_train)):
        Training_data_dic.setdefault('sentence', []).append(X_train[i])
        Training_data_dic.setdefault('label', []).append(Y_train[i])
    # creating validation data dictionary
    Validation_data_dic = {}
    for i in range(len(X_val)):
        Validation_data_dic.setdefault('sentence', []).append(X_val[i])
        Validation_data_dic.setdefault('label', []).append(Y_val[i])
    # creating test data dictionary
    Test_data_dic = {}
    for i in range(len(Test_data_text)):
        Test_data_dic.setdefault('sentence', []).append(Test_data_text[i])
        
    
    # creating vocabulary dictionary
    word_count = 0
    vocab_list = {}
    
    #Create vocab set 
    vocab_set = set()
    for sentence in Training_data_dic['sentence']:
        word_list = tokenize(sentence, gram)
        word_list_reduced = []
        for i in range(len(word_list)):
          #  if word_list[i] not in vocab_set:
                word_list_reduced.append(word_list[i])
        vocab_set.update(word_list_reduced)
    
    #Assign each word a unique index
    for word in vocab_set:
        vocab_list[word] = word_count
        word_count += 1
    
    df_train = Training_data_dic
    df_val = Validation_data_dic
    df_test = Test_data_dic
    
    return vocab_list, df_train, df_val, df_test



