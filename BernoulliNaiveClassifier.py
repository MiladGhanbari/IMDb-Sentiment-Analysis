import numpy as np
# Creating a class for bernoulli naive bayse classifier
class BernoulliNaiveBayse():
    def __init__(self):
        print('Bernoulli Naive Bayse Classifier')
    # function for training of bernoulli naive bayse classification and predicting output of the test data
    def BNB_Classifier(self, X_train, Y_train, X_test):
        Smooth = 1
        NumFeat = X_train[0].size # number of features in the given train data
        UM=[]
        ClassPr = []
        FeatPr = []
        LenM = Y_train.size # length of measurment vector, Y
        UM.append(np.unique(Y_train))
        LenF = X_train[0].size # length of feature vector, X
        NumClass = int(UM[0].size) # number of classes in the given train data 
        Cfeats = {} # defining a dictionary for count of each feature
        Cclass = {} # defining a dictionary for count of each class    
        # constructing count of each features matrix then obtaining class probability and features probability
        for ind in range(LenM):
            if Y_train[ind] not in Cfeats:
                Cfeats[Y_train[ind]] = [0 for j in range (LenF)]
        for ind in range(LenM):
            for con in range(LenF):
                Cfeats[Y_train[ind]][con] += X_train[ind][con]
        for ind in range(LenM):
            if Y_train[ind] in Cclass:
                Cclass[Y_train[ind]] += 1
            else:
                Cclass[Y_train[ind]] = 1
        for CN in  Cfeats:	
            YtrainSize=int(LenM)
            temp = np.array([])
            ClassPr.append(float((float((Cclass[CN] + Smooth)))/(float((YtrainSize + (NumClass * Smooth)))))) 
            for i in range(LenF):
                temp = np.append(temp , float(((Cfeats[CN][i] + Smooth))/float((Cclass[CN]+(2*Smooth)))))
            FeatPr.append(temp)
        PC = ClassPr
        PF = FeatPr
        NC = NumClass
        NF = NumFeat
        Output = np.array([])
        for i in range(X_test.shape[0]):
            ClassID = 0 # ID of Class
            prob_max = -10**10 # maximum probability
            prob = 0 #  probability of each working feature
            for CN in range(NC):
                prob = np.log(PC[CN])
                for j in range(NF):
                    Curent_Class_ID = X_test[i][j]
                    if(Curent_Class_ID == 0):
                        prob += np.log(1-PF[CN][j])
                    else:
                        prob += np.log(PF[CN][j])
                if(prob > prob_max):
                    prob_max = prob
                    ClassID = CN
            Output = np.append(Output , ClassID)
        return Output

    
    
                
