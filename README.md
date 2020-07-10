# IMDb-Sentiment-Analysis
This project studies and compares the performance of several machine learning approaches on sentiment analysis of movie reviews. This project studies and compares the performance of several machine learning approaches on sentiment analysis of movie reviews. An online movie review dataset from the Internet Movie Database (IMDb) website was used as the case study of the project. Several combinations of N-grams, TF * IDF, Bing Liu's lexicon and VADER lexicon features were extracted and used with different classifiers. Furthermore, a dimensionality reduction technique was performed in order to remove the insignificant and redundant extracted features. Finally, several machine learning based classification techniques such as (Bernoulli) Naive Bayes (NB),  Multinomial Naive Bayes (MNB), Support Vector Machines (SVM), the fusion of NB and SVM (NBSVM), Logistic Regression (LR) and two variations of ensemble learning methods were explored for the sentiment analysis task. The optimum values for each of the hyper parameters related to the features and the classifiers were obtained by using grid search approach. In order to build a reliable model comparison scenario and to obtain a robust model, K-fold cross-validation method was conducted on the training data. Based on the extensive experiments on the mentioned models, our best proposed model is a linear SVM, with C = 1000 using top 3M  normalized N-gram (N = 1,2,3) and TF * IDF features which resulted to 90.57 % accuracy on the test set. 

# Instructions:

Libraries Used: Scikit-Learn, NLTK, OS, Numpy, pickle

Directories:
****Put train and test files in the same directory as the codes.****

1. Bernoulli Naive Bayes:
Uses two files: BernoulliNaiveClassifier.py as the algorithm class and NB_main.py as the main file. Run the latter for results.

2. NBSVM a novel combination of SVM and Naive Bayes. NBSVMpreprocessing.py is a function file which is used in the NBSVM.py. Run the latter for results.

3. Using Bing Luis Lexicon Features: Bing_Luis_sentiment.py implements SVM using the aforementioned lexicon. run for results.

4. pipeline_features.py is the comparision between two pipelines for two different methods of feature extraction.

5. SVM.py is our best and final model producing the results stated in the report.

6. VADER.py implements the sentiment analysis function of the NLTK library. run for results.

7. Ensemble.py implements the ensemble method by using voting classifier. run for results.


