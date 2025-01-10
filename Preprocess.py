import os
import time
import re, string, unicodedata
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer
from textblob import TextBlob
from scipy import stats
import time
from time import sleep


def Noise_Removal():
    time.sleep(1)
    print("\nRemoving Hyperlinks, Twitter Marks, Styles, Removing Stop Words, Punctuations,")
    time.sleep(3)
    inputfilename = "data\\resultcnn\\inputdata.csv"
    file_name1='data\\resultcnn\\Noise_Removal.txt'
    file_name2='data\\resultrnn\\Noise_Removal.txt'
    if os.path.exists(file_name1):
        os.remove(file_name1)
    
    with open(inputfilename, 'rt',encoding='utf-8') as f:    
        original_file = f.read()
        rowsplit_data = original_file.splitlines()
        
        for row in  rowsplit_data:            
            a_string = row
            ipstr = a_string.split(",") 
            print("original Text:- ",ipstr[1])
            res=ipstr[1].lower()
            print("After Removal of Twitter Marks, Styles:- ",res)
            res=remove_URL(res)
            print("After URL Remove:- ",res)

            Punctuationsres = re.sub(r'[|$|?|.|!]',r'',res)
            print("After Punctuations removal ",Punctuationsres)

            Hashtagres = ''
            for i in Punctuationsres.split():
                 if i[:1] == '@':
                     pass
                 elif i[:1] == '#':                     
                     pass
                 else:
                     Hashtagres = Hashtagres.strip() + ' ' + i

            print("After Hashtag Removal ",Hashtagres)
            stop_words = set(stopwords.words('english')) 
            word_tokens = word_tokenize(Hashtagres) 
            filtered_sentence = [w for w in word_tokens if not w in stop_words] 
            filtered_sentence = []
            stopres=""
            for w in word_tokens: 
                if w not in stop_words:
                    stopres=stopres+w+" "
                    filtered_sentence.append(w) 
            print("After Stop word Remove:- ",stopres)
            str=stopres
            file = open(file_name1, "a")            
            file.write(str)
            file.write("\n")    
            file.close()

    if os.path.exists(file_name2):
        os.remove(file_name2)
        
        with open(inputfilename, 'rt',encoding='utf-8') as f:    
            original_file = f.read()
            rowsplit_data = original_file.splitlines()
                
            for row in  rowsplit_data:            
                a_string = row
                ipstr = a_string.split(",") 
                print("original Text:- ",ipstr[1])
                res=ipstr[1].lower()
                print("After Removal of Twitter Marks, Styles:- ",res)
                res=remove_URL(res)
                print("After URL Remove:- ",res)

                Punctuationsres = re.sub(r'[|$|?|.|!]',r'',res)
                print("After Punctuations removal ",Punctuationsres)

                Hashtagres = ''
                for i in Punctuationsres.split():
                        if i[:1] == '@':
                            pass
                        elif i[:1] == '#':                     
                            pass
                        else:
                            Hashtagres = Hashtagres.strip() + ' ' + i

                print("After Hashtag Removal ",Hashtagres)
                stop_words = set(stopwords.words('english')) 
                word_tokens = word_tokenize(Hashtagres) 
                filtered_sentence = [w for w in word_tokens if not w in stop_words] 
                filtered_sentence = []
                stopres=""
                for w in word_tokens: 
                    if w not in stop_words:
                        stopres=stopres+w+" "
                        filtered_sentence.append(w) 
                print("After Stop word Remove:- ",stopres)
                str=stopres
                file = open(file_name1, "a")            
                file.write(str)
                file.write("\n")    
                file.close()            
    print("\n\t\t100% Noise Removal Process is Completed.\n")
        
def Tokenization():
    file_name1='data\\resultcnn\\Noise_Removal.txt'
    file_name2='data\\resultcnn\\Tokenization.txt'
    file_name3='data\\resultrnn\\Tokenization.txt'
    if os.path.exists(file_name2):
        os.remove(file_name2)
    file = open(file_name1, "r")
    file2 = open(file_name2, "a")  
    Lines = file.readlines()
    for line in Lines:
        print("Before Tokenization - ",line)
        textBlb = TextBlob(line)          
        tokens = nltk.sent_tokenize(line)
        print ("After Tokenization - ",tokens)
        file2.write(str(tokens))
        file2.write("\n")
    file2.close()
    file.close()

    if os.path.exists(file_name3):
        os.remove(file_name3)
    file = open(file_name1, "r")
    file3 = open(file_name3, "a")  
    Lines = file.readlines()
    for line in Lines:
        print("Before Tokenization - ",line)
        textBlb = TextBlob(line)          
        tokens = nltk.sent_tokenize(line)
        print ("After Tokenization - ",tokens)
        file3.write(str(tokens))
        file3.write("\n")
    file3.close()
    file.close()
    
    print("\n\t\t100% Tokenization Process Completed.\n")

def Normalization():
    file_name1='data\\resultcnn\\Tokenization.txt'
    file_name2='data\\resultcnn\\preprocessing.txt'
    file_name3='data\\resultrnn\\preprocessing.txt'
    if os.path.exists(file_name2):
        os.remove(file_name2)
    file = open(file_name1, "r")
    file2 = open(file_name2, "a")  
    Lines = file.readlines()
    for line in Lines:
        print("Before Normalization  - ",line)
        
        ps = PorterStemmer()
        Stemmingres=""
        inputstr = line.split(" ")
        for w in inputstr:
            Stemmingres=Stemmingres+ps.stem(w)+" "
        pos_Taggingres=word_tokenize(Stemmingres)
        print ("After Normalization  - ",str(pos_Taggingres))
        file2.write(str(pos_Taggingres))
        file2.write("\n")
    file2.close()
    file.close()
    
    if os.path.exists(file_name3):
        os.remove(file_name2)
    file = open(file_name1, "r")
    file3 = open(file_name2, "a")  
    Lines = file.readlines()
    for line in Lines:
        print("Before Normalization  - ",line)
        
        ps = PorterStemmer()
        Stemmingres=""
        inputstr = line.split(" ")
        for w in inputstr:
            Stemmingres=Stemmingres+ps.stem(w)+" "
        pos_Taggingres=word_tokenize(Stemmingres)
        print ("After Normalization  - ",str(pos_Taggingres))
        file3.write(str(pos_Taggingres))
        file3.write("\n")
    file3.close()
    file.close()
    print("\n\t\t100% Pre-Processing Process Completed.\n")

def convert_emoticons(text):    
    for emot in EMOTICONS:
        text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)    
    return text

def Hashtag_Removal(text):
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

def remove_URL(urlstr):    
    return re.sub(r"http\S+", "", urlstr)

def remove_stopwords(words):    
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def Dec_scale(trans):
    for x in trans:
        p = trans[x].max()
        q = len(str(abs(p)))
        trans[x] = trans[x]/10**q 

def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])
