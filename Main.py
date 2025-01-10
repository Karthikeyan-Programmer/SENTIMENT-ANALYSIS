import os
import sys
import csv
import shutil
import time
import warnings
import pandas as pd
from pyspark import SparkContext
from tkinter import *
import time
from time import sleep
from Preprocess import *
from Classification_CNN import *
from Classification_RNN import *
from WordCloud import *
from Twitter_Sentiment_Analysis import *
from Metrics import *
from tkinter import messagebox

def Loadthetweets():
    print ("\t\t\t |--------- ****** ANALYSIS OF PUBLIC SENTIMENT ON RETURN TO OFFICE ****** --------|")
    time.sleep(3)
    warnings.filterwarnings('ignore')    
    file_name='data\\\dataset\\dataset.csv'
    time.sleep(1)
    print('==========================================================================================')
    print ("\t\t\t ****** LOADING THE DATASET ******")
    print('==========================================================================================')
    time.sleep(1)
    print('\nLoading Dataset process is starting\n')
    time.sleep(1)   
    with open(file_name, 'rt',encoding='utf-8') as f:    
        original_file = f.read()
        rowsplit_data = original_file.splitlines()                
        for row in  rowsplit_data:
            print(row)            
            df = pd.read_csv(file_name, sep=",")
            df = df.drop(df.columns[0], axis=1)                       
            df1= df.iloc[:,[8]]
            df2= df.iloc[:,[8]] 
            df1.to_csv('data\\resultcnn\\inputdata.csv', header= True)
            df2.to_csv('data\\resultrnn\\inputdata.csv', header= True)
            print()
    time.sleep(1)
    print('\nLoading Dataset process is Completed...!\n')        
    print('==========================================================================================')
    time.sleep(1)
    messagebox.showinfo('Dataset', 'Loading Dataset process is Completed!')
    time.sleep(1)
    inputfilename = "data\\resultcnn\\inputdata.csv"    
    print ("\t\t\t ****** SELECTED TWITTER TEXT ******")
    print('==========================================================================================')
    print('\nSelected Twitter Text process is starting\n')
    time.sleep(3)
    with open(inputfilename, 'rt',encoding='utf-8') as f:    
        original_file = f.read()
        rowsplit_data = original_file.splitlines()        
        for row in  rowsplit_data:            
            a_string = row
            ipstr = a_string.split(",")            
            print(ipstr[1])
            print()
    print('\nSelected Twitter Text process is Completed...!\n')
    print('==========================================================================================')
    messagebox.showinfo('Twitter', 'Selected Twitter Text process is Completed!')
    time.sleep(1)
    print('\nNext Click PREPROCESSING button\n')

def Preprocessing():
    time.sleep(1)
    print('==========================================================================================')
    print ("\t\t\t ****** DATA PREPROCESSING ******")    
    print('==========================================================================================')
    print('\nNoice Removal process is starting...\n')    
    sleep(1)
    Noise_Removal();
    print('\nNoice Removal process is Completed!\n')
    print('==========================================================================================')
    messagebox.showinfo('Noice', 'Noice Removal process is Completed...!')
    time.sleep(1)
    print('\nTokenization process is starting\n')    
    time.sleep(1)
    Tokenization();
    print('\nTokenization process is Completed!\n')
    print('==========================================================================================')
    messagebox.showinfo('Tokenization', 'Tokenization process is Completed...!')
    time.sleep(1)
    print('\nNormalization process is starting\n')    
    time.sleep(1)
    Normalization();
    print('\nNormalization process is Completed!\n')
    print('==========================================================================================')
    messagebox.showinfo('Normalization', 'Normalization process is Completed...!')
    time.sleep(1)
    print('\nNext Click SENTIMENT CLASSIFICATION button\n')

def sentimentclassificationCNN():
    time.sleep(1)
    print ("\t\t\t ****** SENTIMENT CLASSIFICATION ******")        
    print('==========================================================================================')
    print('\nSentiment Classification Using CNN process is starting\n')
    time.sleep(3)
    sentiment_classificationCNN();
    time.sleep(1)
    print('\nSentiment Classification Using CNN process is Completed...!\n')
    print('==========================================================================================')
    messagebox.showinfo('Sentiment Classification', 'Sentiment Classification process is Completed!')
    time.sleep(1)
    print('\nNext Click FEATURE EXTRACTION button\n')

def sentimentclassificationRNN():
    time.sleep(1)
    print ("\t\t\t ****** SENTIMENT CLASSIFICATION ******")        
    print('==========================================================================================')
    print('\nSentiment Classification Using RNN process is starting\n')
    time.sleep(3)
    sentiment_classificationRNN();
    time.sleep(1)
    print('\nSentiment Classification Using RNN process is Completed...!\n')
    print('==========================================================================================')
    messagebox.showinfo('Sentiment Classification', 'Sentiment Classification process is Completed!')
    time.sleep(1)
    print('\nNext Click FEATURE EXTRACTION button\n')

def FeatureExtraction():
    time.sleep(1)
    print ("\t\t\t ****** FEATURE EXTRACTION ******")        
    print('==========================================================================================')
    print('\nExtract the features from preprocessing process is starting\n')    
    time.sleep(3)
    Feature_Extract(); 
    print('\nExtract the features from preprocessing process is Completed...!\n')
    print('==========================================================================================')
    messagebox.showinfo('Extract Features', 'Extract the features from preprocessing process is Completed!')
    time.sleep(1)
    print('\nNext Click FEATURE CLASSIFICATION button\n')

def FeatureClassificationcnn():
    time.sleep(1)
    print ("\t\t\t ****** FEATURE CLASSIFICATION ******")
    print('==========================================================================================')
    print('\nFeatures Classification process is starting\n')    
    time.sleep(1)
    Feature_Classification()
    time.sleep(1)
    print('\nFeatures Classification process is Completed...!\n')
    print('==========================================================================================')
    messagebox.showinfo('Features', 'Features Classification process is Completed!')      
    time.sleep(3)
    '''visualizing via word cloud what buzz words'''
    print("\nvisualizing via word cloud what buzz words\n")

    dataset = pd.read_csv('data\\resultcnn\\inputdata.csv', encoding = 'ISO-8859-1')
    dataset = pd.read_csv('data\\resultrnn\\inputdata.csv', encoding = 'ISO-8859-1')

    #print(dataset.head())

    def gen_freq(text):
        #Will store the list of words
        word_list = []

        #Loop over all the tweets and extract words into word_list
        for tw_words in text.split():
            word_list.extend(tw_words)

        #Create word frequencies using word_list
        word_freq = pd.Series(word_list).value_counts()

        #Print top 20 words
        word_freq[:20]
        
        return word_freq

    #gen_freq(dataset.text.str)
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    #Generate word frequencies
    word_freq = gen_freq(dataset.text.str)

    #Generate word cloud
    wc = WordCloud(width=400, height=330, max_words=100, background_color='white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(12, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    ##### cleaning task, removing useless words and letters as visible from
    # the wordcloud above
    import re

    def clean_text(text):
        #Remove RT
        text = re.sub(r'RT', '', text)
        
        #Fix &
        text = re.sub(r'&amp;', '&', text)
        
        #Remove punctuations
        text = re.sub(r'[?!.;:,#@-]', '', text)

        #Convert to lowercase to maintain consistency
        text = text.lower()
        return text
    from wordcloud import STOPWORDS
    text = dataset.text.apply(lambda x: clean_text(x))
    word_freq = gen_freq(text.str)*100
    word_freq = word_freq.drop(labels=STOPWORDS, errors='ignore')

    #Generate word cloud
    wc = WordCloud(width=450, height=330, max_words=200, background_color='white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(12, 14))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    time.sleep(1)
    wordcloudcnn.generate();
    time.sleep(1)
    print('\nNext Click PERFORMANCE METRICS button\n')

def FeatureClassificationrnn():
    time.sleep(1)
    print ("\t\t\t ****** FEATURE CLASSIFICATION ******")
    print('==========================================================================================')
    print('\nFeatures Classification process is starting\n')    
    time.sleep(1)
    Feature_Classification()
    time.sleep(1)
    print('\nFeatures Classification process is Completed...!\n')
    print('==========================================================================================')
    messagebox.showinfo('Features', 'Features Classification process is Completed!')      
    time.sleep(3)
    '''visualizing via word cloud what buzz words'''
    print("\nvisualizing via word cloud what buzz words\n")

    dataset = pd.read_csv('data\\resultcnn\\inputdata.csv', encoding = 'ISO-8859-1')
    dataset = pd.read_csv('data\\resultrnn\\inputdata.csv', encoding = 'ISO-8859-1')

    #print(dataset.head())

    def gen_freq(text):
        #Will store the list of words
        word_list = []

        #Loop over all the tweets and extract words into word_list
        for tw_words in text.split():
            word_list.extend(tw_words)

        #Create word frequencies using word_list
        word_freq = pd.Series(word_list).value_counts()

        #Print top 20 words
        word_freq[:20]
        
        return word_freq

    #gen_freq(dataset.text.str)
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    #Generate word frequencies
    word_freq = gen_freq(dataset.text.str)

    #Generate word cloud
    wc = WordCloud(width=400, height=330, max_words=100, background_color='white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(12, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    ##### cleaning task, removing useless words and letters as visible from
    # the wordcloud above
    import re

    def clean_text(text):
        #Remove RT
        text = re.sub(r'RT', '', text)
        
        #Fix &
        text = re.sub(r'&amp;', '&', text)
        
        #Remove punctuations
        text = re.sub(r'[?!.;:,#@-]', '', text)

        #Convert to lowercase to maintain consistency
        text = text.lower()
        return text
    from wordcloud import STOPWORDS
    text = dataset.text.apply(lambda x: clean_text(x))
    word_freq = gen_freq(text.str)*100
    word_freq = word_freq.drop(labels=STOPWORDS, errors='ignore')

    #Generate word cloud
    wc = WordCloud(width=450, height=330, max_words=200, background_color='white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(12, 14))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    time.sleep(1)
    wordcloudrnn.generate();
    time.sleep(1)
    compare();
    print('\nNext Click PERFORMANCE METRICS button\n')

def PerformanceMetricscnn():
    time.sleep(1)
    print('\nGraph generation process is starting\n')        
    time.sleep(3)
    Processcnn();
    print('\nGraph is Generated Successfully...!')
    print('==========================================================================================')
    
def PerformanceMetricsrnn():
    time.sleep(1)
    print('\nGraph generation process is starting\n')        
    time.sleep(3)
    Processrnn();
    print('\nGraph is Generated Successfully...!')
    print('==========================================================================================')
      
def main_screen():    
    global window
    window = Tk()
    window.geometry("600x330")    
    window.title("ANALYSIS OF PUBLIC SENTIMENT ON RETURN TO OFFICE")
    window['background']='#ED9121'
    Label(window, text = "ANALYSIS OF PUBLIC SENTIMENT ON RETURN TO OFFICE",bg = "#9932CC",fg ="yellow",width = "400", height = "4",font=('Times New Roman Bold',12)).pack()
    Label(text = "",height=1).pack()
    Label(text = "").pack()
    Button(text = "START SIMULATION", height = "2", width = "40",fg ="BLUE",font=('Times New Roman Bold',14), command = Loadthetweets).pack()
    Label(text = "").pack()
    Button(text = "PREPROCESSING", height = "2", width = "40",fg ="BLUE",font=('Times New Roman Bold',14), command = Preprocessing).pack()
    Label(text = "").pack()
    Button(text = "SENTIMENT CLASSIFICATION USING CNN", height = "2", width = "40",fg ="BLUE",font=('Times New Roman Bold',14), command = sentimentclassificationCNN).pack()
    Label(text = "").pack()
    Button(text = "FEATURE EXTRACTION", height = "2", width = "40",fg ="BLUE",font=('Times New Roman Bold',14), command = FeatureExtraction).pack()
    Label(text = "").pack()
    Button(text = "FEATURE CLASSIFICATION", height = "2", width = "40",fg ="BLUE",font=('Times New Roman Bold',14), command = FeatureClassificationcnn).pack()
    Label(text = "").pack()
    Button(text = "PERFORMANCE METRICS", height = "2", width = "40",fg ="BLUE",font=('Times New Roman Bold',14), command = PerformanceMetricscnn).pack()
    Label(text = "").pack()
    window.mainloop()
    global window1
    window1 = Tk()
    window1.geometry("600x330")    
    window1.title("ANALYSIS OF PUBLIC SENTIMENT ON RETURN TO OFFICE")
    window1['background']='green'
    Label(window1, text = "ANALYSIS OF PUBLIC SENTIMENT ON RETURN TO OFFICE",bg = "purple",fg ="yellow",width = "400", height = "4",font=('Times New Roman Bold',12)).pack()
    Label(text = "",height=1).pack()
    Label(text = "").pack()
    Button(text = "START SIMULATION", height = "2", width = "40",fg ="BLUE",font=('Times New Roman Bold',14), command = Loadthetweets).pack()
    Label(text = "").pack()
    Button(text = "PREPROCESSING", height = "2", width = "40",fg ="BLUE",font=('Times New Roman Bold',14), command = Preprocessing).pack()
    Label(text = "").pack()
    Button(text = "SENTIMENT CLASSIFICATION USING RNN", height = "2", width = "40",fg ="BLUE",font=('Times New Roman Bold',14), command = sentimentclassificationRNN).pack()
    Label(text = "").pack()
    Button(text = "FEATURE EXTRACTION", height = "2", width = "40",fg ="BLUE",font=('Times New Roman Bold',14), command = FeatureExtraction).pack()
    Label(text = "").pack()
    Button(text = "FEATURE CLASSIFICATION", height = "2", width = "40",fg ="BLUE",font=('Times New Roman Bold',14), command = FeatureClassificationrnn).pack()
    Label(text = "").pack()
    Button(text = "PERFORMANCE METRICS", height = "2", width = "40",fg ="BLUE",font=('Times New Roman Bold',14), command = PerformanceMetricsrnn).pack()
    Label(text = "").pack()
    window.mainloop()
main_screen()
