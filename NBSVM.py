from NBSVMpreprocessing import create_bow, build_vocab
import numpy as np
from sklearn.svm import LinearSVC
import time
"""
Naive Bayes Support Vector Machine interpolation, NBSVM.
"""

## tuning_params:
gram = 2
C = 100
beta = 0.25
alpha = 1


"""
Trains the Multinomial Naive Bayes Model
"""
def train_nb(vocab_list, df):
    
    #find prior = total positive examples/total examples 
    total_sents = len(df['label'])
    pos_sents = 0
    neg_sents = 0
    for i in range(len(df['label'])):
        if(df['label'][i] == 1):
            pos_sents += 1
    neg_sents = total_sents - pos_sents
    
    #initiate counts for word appearance conditional on label == 1 and label == 0
    #alpha is laplacian smoothing parameter
    pos_list = np.ones(len(vocab_list)) * alpha
    neg_list = np.ones(len(vocab_list)) * alpha
    
    for sentence, label in zip(df['sentence'], df['label']):
        bow = create_bow(sentence, vocab_list, gram)
      
        if label == 1:
            pos_list += bow
        else:
            neg_list += bow
            
    #Calculate log-count ratio
    x = (pos_list/abs(pos_list).sum())
    y = (neg_list/abs(neg_list).sum())
    r = np.log(x/y)
    b = np.log(pos_sents/neg_sents)
    
    return r, b

"""
Trains the (linear-kernel) SVM with L2 Regularization
"""
def train_svm(vocab_list, df_train, c, r):
#    clf = LinearSVC(C=c, class_weight=None, dual=False, fit_intercept=True,
#     loss='squared_hinge', max_iter=1000,
#     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
#     verbose=0)
    print('creating SVM model')
    clf = LinearSVC(C=c)
    print('creating training matrix')
    M = np.array([])
    X = np.zeros((len(df_train['sentence']), len(vocab_list)))
    bow = np.array([])
    con = 0
    for sentence in df_train['sentence']:
        print('iteration: {}'.format(con+1))
        bow = create_bow(sentence, vocab_list, gram)
        M = r * bow
        for i in range(len(M)):
            X[con, i] = M[i]
#       X.append(M)
        con=con+1
    #X = np.array([(r * create_bow(sentence, vocab_list, gram))  for sentence in df_train['sentence']])
    y = df_train['label']
   
    clf.fit(X, y)   
    svm_coef = clf.coef_
    svm_intercept = clf.intercept_
    
    return svm_coef, svm_intercept, clf

"""
Predict classification with MNB
"""
def predict(df_test, w, b, vocab_list):
    total_sents = len(df_test['label'])
    total_score = 0
    
    for sentence, label in zip(df_test['sentence'], df_test['label']):      
        bow = create_bow(sentence, vocab_list, gram)

        result = np.sign(np.dot(bow, w.T) + b)
        if result == -1:
            result = 0
        if result == label:
            total_score +=1      
            
    return total_score/total_sents

"""
Predict classification with NB-SVM
"""
def predict_nbsvm(df_test, svm_coef, svm_intercept, r, b, vocab_list):
    total_sents = len(df_test['label'])
    total_score = 0
    
    for sentence, label in zip(df_test['sentence'], df_test['label']):
        bow = r * create_bow(sentence, vocab_list, gram)  
        w_bar = (abs(svm_coef).sum())/len(vocab_list)
        w_prime = (1 - beta)*(w_bar) + (beta * svm_coef)
        result = np.sign(np.dot(bow, w_prime.T) + svm_intercept)
        if result == -1:
            result = 0
        if result == label:
            total_score +=1  
            
    return total_score/total_sents


    
if __name__ == "__main__":
    
    time_first = time.time()
    print("Building Dataset...")
    vocab_list, df_train, df_val, df_test = build_vocab(gram)

      
    print("Training Multinomial Naive Bayes...")
    r, b = train_nb(vocab_list, df_train)

    #Train SVM
    print("Training LinearSVM...")
    svm_coef, svm_intercept, clf = train_svm(vocab_list, df_train, C, r)
   
  
    #Test Models
    print("Test using NBSVM ({:.4f}-gram):".format(gram))
    accuracy = predict_nbsvm(df_val, svm_coef, svm_intercept, r, b, vocab_list)
    print("Beta: {} Accuracy: {}".format(beta, accuracy))
    
    print("Test using MNB ({:.4f}-gram):".format(gram))
    mnb_acc = predict(df_val, r, b, vocab_list)
    print("Accuracy: {}".format(mnb_acc))
    
    


    



    

    
            
            
