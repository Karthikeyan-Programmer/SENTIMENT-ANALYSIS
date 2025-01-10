#################
#### imports ####
#################
import pandas as pd
## Importing stopwords
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import string
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


def compare():
        print("\n|----------- Compare the results for these two networks -----------|\n")

        def list_files(directory):
                allfiles = []	
                for dirname, dirnames, filenames in os.walk(directory):
                        for filename in filenames:
                                allfiles.append(os.path.join(dirname,filename))
                return allfiles

        def SentimentsCounter(sentiment):
                if(str(sentiment)=="['negative']"):
                        global Negative_Count
                        Negative_Count  +=1
                elif(str(sentiment)=="['positive']"):
                        global Positive_Count
                        Positive_Count +=1
                elif (str(sentiment)=="['neutral']"):
                        global Neutral_Count
                        Neutral_Count += 1

        # ## Loading Data
        train_data = pd.read_csv("data\\resultcnn\\x_y_train.csv")
        test_data = pd.read_csv("data\\resultcnn\\x_test.csv")

        ## Training data
        x_train_data = train_data["text"]
        y_train_data = train_data["sentiment"]

        ## Testing data
        x_test_data = test_data["text"]


        ########################
        #### Stop Words ########
        ########################
        stop = stopwords.words("english")
        ## Importing punctuations
        punctuations = string.punctuation
        ## Adding Punctuations to our stop words
        stop += punctuations



        ##########################
        #### Train and Test Split ####
        ##########################

        ## Splitting in the given training data for our training and testing
        x_train_train, x_train_test, y_train_train, y_train_test = train_test_split(x_train_data, y_train_data, 
                                                                                    random_state = 0, test_size = 0.25)

        ##########################
        #### Count Vectoriser ####
        ##########################
        from sklearn.feature_extraction.text import TfidfVectorizer
        tf_idf_vec = TfidfVectorizer(max_features=3000, ngram_range=(1,2), stop_words=stop, 
                                     analyzer='word', max_df = 0.8, lowercase = True, use_idf = True, smooth_idf = True)

        ## Fit transform the training data
        train_features = tf_idf_vec.fit_transform(x_train_train)

        ## Only transform the testing data according to the features which was fit using x_train
        test_features = tf_idf_vec.transform(x_train_test)

        ######################################
        ######################################
        #### Applying various classifiers ####
        ######################################
        ######################################

        ###########################################################
        ####  Applying RNN (Recurrent Neural Network) #############
        ###########################################################
        from sklearn.svm import SVC
        rnn = SVC()
        rnn.fit(train_features, y_train_train)

        print("RNN (Recurrent Neural Network)")
        print(rnn.score(test_features, y_train_test))


        ##################################################
        #### Applying CNN (Convoluted Neural Network) ####
        ##################################################

        from sklearn.naive_bayes import MultinomialNB
        cnn = MultinomialNB(alpha=0.4)
        cnn.fit(train_features, y_train_train)

        print("CNN (Convoluted Neural Network)")
        print(cnn.score(test_features, y_train_test))

        pickle.dump(cnn, open(r"ML_Model_Sentiment_Analysis.pkl", "wb"))



