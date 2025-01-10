from __future__ import print_function
import sys
import os
import time
import glob
import pytest
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import text2emotion as te

import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tkinter import *
from scipy import stats
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense
from st_optics import ST_OPTICS
from catboost import Pool, CatBoostClassifier
from layers import SelfAttention
from agent import Agent

sys.path.append('..')  

model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    random_strength=0.1,
    depth=8,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    leaf_estimation_method='multi_emotion_classification'
)

train_data=pd.read_csv('data\\resultrnn\\inputdata.csv')
parser = argparse.ArgumentParser()
parser.add_argument("--config", default=0,
                    help="Integer value representing a model configuration")
args = parser.parse_args()
tf.random.set_seed(100)
vocabulary_size = 1000
sequence_length = 500  
embedding_dims = 50    
batch_size = 100       
num_epochs = 10        
config = int(args.config) 
X = Input(shape=(sequence_length,), batch_size=batch_size)
embedded = Embedding(input_dim=vocabulary_size, output_dim=embedding_dims)(X)
if config == 1:
    embedded, attention_weights = SelfAttention(size=50,
                                                num_hops=6,
                                                use_penalization=False)(embedded)
elif config == 2:
    embedded, attention_weights = SelfAttention(size=50,
                                                num_hops=6,
                                                use_penalization=True,
                                                penalty_coefficient=0.1)(embedded)
embedded_flattened = Flatten()(embedded)
fully_connected = Dense(units=250, activation='relu')(embedded_flattened)
Y = Dense(units=1, activation='sigmoid')(fully_connected)


def sentiment_classificationRNN():
    time.sleep(1)
    print("\n\t\tsentiment classification using RNN (Recurrent Neural Network)")
    time.sleep(3)
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=1000, output_dim=64))

    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
    model.add(layers.GRU(256, return_sequences=True))

    # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
    model.add(layers.SimpleRNN(128))

    model.add(layers.Dense(10))

    model.summary()

    encoder_vocab = 1000
    decoder_vocab = 2000

    encoder_input = layers.Input(shape=(None,))
    encoder_embedded = layers.Embedding(input_dim=encoder_vocab, output_dim=64)(
        encoder_input
    )

    # Return states in addition to output
    output, state_h, state_c = layers.LSTM(64, return_state=True, name="encoder")(
        encoder_embedded
    )
    encoder_state = [state_h, state_c]

    decoder_input = layers.Input(shape=(None,))
    decoder_embedded = layers.Embedding(input_dim=decoder_vocab, output_dim=64)(
        decoder_input
    )

    # Pass the 2 states to a new LSTM layer, as initial state
    decoder_output = layers.LSTM(64, name="decoder")(
        decoder_embedded, initial_state=encoder_state
    )
    output = layers.Dense(10)(decoder_output)

    model = keras.Model([encoder_input, decoder_input], output)
    model.summary()
    time.sleep(3)
    file_name1='data\\resultrnn\\preprocessing.txt'
    file = open(file_name1, "r")
    Lines = file.readlines()
    for line in Lines:
        print("Pre-Processing Data - ",line)
        ipstr = line.split("'")
        res1=""
        for row in  ipstr:            
            res1 = res1+row
        res2=res1.split("[")
        res3=""
        for row in  res2:            
            res3 = res3+row
        res4=res3.split("]")
        res=""
        for row in  res4:            
            res = res+row         
        emotionres=te.get_emotion(res)
        resstr=str(emotionres)
        happy=resstr[resstr.index('Happy')+8:resstr.index('Angry')-3]
        angry=resstr[resstr.index('Angry')+8:resstr.index('Surprise')-3]
        surprise=resstr[resstr.index('Surprise')+11:resstr.index('Sad')-3]
        sad=resstr[resstr.index('Sad')+6:resstr.index('Fear')-3]
        fear=resstr[resstr.index('Fear')+7:resstr.index('}')]
        trust=(float(happy)+float(angry)+float(surprise)+float(sad)+float(fear))/5
        if (((float(happy)>0.0) and (float(surprise) >0.0)) and ((float(angry) == 0.0) and (float(sad) == 0.0)and (float(fear) == 0.0))):
            print("Sentiment Classification Result = POSITIVE\n")
        elif (((float(happy)==0.0) and (float(surprise) ==0.0)) and ((float(angry)>0.0) or (float(sad)>0.0)or (float(fear)>0.0))):
            print("Sentiment Classification Result =  NEGATIVE\n")
        else:
            print("Sentiment Classification Result =  NEUTRAL\n")
    file.close()
    print("\n\t\t100% Sentiment Classification Process Completed.")
    
st_optics = ST_OPTICS(xi = 0.4, eps2 = 100, min_samples = 5)

def Feature_Classification():    
    file_name1='data\\resultrnn\\preprocessing.txt'
    f_name1='data\\resultrnn\\Positive.txt'
    if os.path.exists(f_name1):
        os.remove(f_name1)
    f_name2='data\\resultrnn\\Negative.txt'
    if os.path.exists(f_name2):
        os.remove(f_name2)
    f_name3='data\\resultrnn\\Neutral.txt'
    if os.path.exists(f_name3):
        os.remove(f_name3)
    file = open(file_name1, "r")
    file1 = open(f_name1, "a")
    file2 = open(f_name2, "a")
    file3 = open(f_name3, "a")
    Lines = file.readlines()    
    for line in Lines:        
        ipstr = line.split("'")
        res1=""
        for row in  ipstr:            
            res1 = res1+row
        res2=res1.split("[")
        res3=""
        for row in  res2:            
            res3 = res3+row

        res4=res3.split("]")
        res=""
        for row in  res4:            
            res = res+row         
        emotionres=te.get_emotion(res)
        resstr=str(emotionres)
        happy=resstr[resstr.index('Happy')+8:resstr.index('Angry')-3]
        angry=resstr[resstr.index('Angry')+8:resstr.index('Surprise')-3]
        surprise=resstr[resstr.index('Surprise')+11:resstr.index('Sad')-3]
        sad=resstr[resstr.index('Sad')+6:resstr.index('Fear')-3]
        fear=resstr[resstr.index('Fear')+7:resstr.index('}')]
        trust=(float(happy)+float(angry)+float(surprise)+float(sad)+float(fear))/5
        
        if (((float(happy)>0.0) and (float(surprise) >0.0)) and ((float(angry) == 0.0) and (float(sad) == 0.0)and (float(fear) == 0.0))):            
            file1.write(res)
            file1.write("\n")
        elif (((float(happy)>0.0) or (float(surprise) >0.0)) and ((float(angry)==0.0) and (float(sad)==0.0)and (float(fear)==0.0))):            
            file2.write(res)
            file2.write("\n")
        elif (((float(happy)==0.0) and (float(surprise) ==0.0)) and ((float(angry)>0.0) and (float(sad)>0.0)and (float(fear)>0.0))):
            file3.write(res)
            file3.write("\n")
        else:            
            file3.write(res)
            file3.write("\n")
    file.close()
    file1.close()
    file2.close()
    file3.close()
    file1 = open(f_name1, "r")
    file2 = open(f_name2, "r")
    file3 = open(f_name3, "r")
   
    print("\n\t\t\t ++++++ POSITIVE ++++++ \n")
    Lines1 = file1.readlines()
    for line in Lines1:
        print(line)
    
    print("\n\t\t\t ------ NEGATIVE ------ \n")
    Lines2 = file2.readlines()
    for line in Lines2:
        print(line)
    
    print("\n\t\t\t @@@@@@ NEUTRAL @@@@@@ \n")
    Lines3 = file3.readlines()
    for line in Lines3:
        print(line)
        
    print("\n\t\t100% Feature Classification Process Completed.")

    
def Feature_Extract():    
    file_name1='data\\resultrnn\\preprocessing.txt'    
    file = open(file_name1, "r")
    Lines = file.readlines()
    for line in Lines:
        print("Before Feature Extraction - ",line)
        ipstr = line.split("'")
        res1=""
        for row in  ipstr:            
            res1 = res1+row
        res2=res1.split("[")
        res3=""
        for row in  res2:            
            res3 = res3+row
        res4=res3.split("]")
        res=""
        for row in  res4:            
            res = res+row         
        emotionres=te.get_emotion(res)
        print ("After Feature Extraction : ",emotionres)
    file.close()
    print("\n\t\t100% Feature Extraction Process Completed.")









