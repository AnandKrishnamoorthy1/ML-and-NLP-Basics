# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 01:53:41 2019

@author: anand
"""

from nltk.tokenize import sent_tokenize
from nltk import word_tokenize,ne_chunk, pos_tag,Tree,ngrams
from nltk.stem import WordNetLemmatizer,PorterStemmer
#Reading Text

#http://www.nltk.org/book/
with open(r"A Short History of Computers and Computing.txt","r") as inp_txt:
    in_txt=inp_txt.readlines()

full_text=''    
for lines in in_txt:
    full_text+=lines
sent=sent_tokenize(full_text)

tokens = word_tokenize(sent[0])
print(word_tokenize("At about the same time (the late 1930's) John Atanasoff of Iowa State University and his assistant Clifford Berry built the first digital computer that worked electronically, the ABC (Atanasoff-Berry Computer)."))
lemma = WordNetLemmatizer()
stem=PorterStemmer()
stem_wrds=[]
lemma_wrds=[]
for token in tokens:
    stem_wrds.extend([stem.stem(token)])
    lemma_wrds.extend([lemma.lemmatize(token)])
    
print(pos_tag(tokens))
print(stem_wrds)
print(lemma_wrds)

sent="John Works in FVDS and stays in Chennai"
tokens = word_tokenize(sent)
chunked=ne_chunk(pos_tag(tokens))
for elt in chunked:
    if isinstance(elt, Tree):
        print(elt)

###############################################################################
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split,cross_val_score
import pandas as pd
from sklearn.metrics import confusion_matrix

inp = pd.read_excel(r"\Movie review.xlsx",encoding='utf-8' )
X=inp.SNTC_TXT
y=inp.REVIEW
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

vect=CountVectorizer().fit(X_train)
X_train_vect=vect.transform(X_train)  

def try_MultinomialNB():
    clf=MultinomialNB().fit(X_train_vect,y_train)
    predict_class=clf.predict(vect.transform(X_test))
    compare_class=pd.DataFrame(data={'Actual Value':y_test,'Predicted Value':predict_class})
    #print(compare_calss.round(2))
    print("Confusion matrix for Count vectorizor and MultinomialNB:")
    print(confusion_matrix(y_test, predict_class))
    
try_MultinomialNB()