# -*- coding: utf-8 -*-
"""
Created on Sat May  5 23:05:35 2018

@author: anand
"""

###############################################################################

# Load the library with the iris dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

iris = load_iris()
# Create a dataframe with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target']=iris.target
df['species']=df['target'].replace({0:'setosa',1:'versicolor',2:'virginica'})

x=df["sepal length (cm)"]
y=df["sepal width (cm)"]
plt.figure()
plt.scatter(x,y)
plt.show()
###############################################################################

# Load the library with the iris dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

iris = load_iris()
# Create a dataframe with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target']=iris.target
df['species']=df['target'].replace({0:'setosa',1:'versicolor',2:'virginica'})

plt.figure()
x=df[df.target==0]
plt.scatter(x["sepal length (cm)"],x["petal length (cm)"],c="green")
y=df[df.target==1]
plt.scatter(y["sepal length (cm)"],y["petal length (cm)"],c="blue")
z=df[df.target==2]
plt.scatter(z["sepal length (cm)"],z["petal length (cm)"],c="yellow")
plt.show()

###############################################################################
#Load the library with the iris dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
# Import KMeans
from sklearn.cluster import KMeans

iris = load_iris()
# Create a dataframe with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target']=iris.target
df['species']=df['target'].replace({0:'setosa',1:'versicolor',2:'virginica'})

X=df[["sepal length (cm)","sepal width (cm)","petal length (cm)"]]
# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)
# Fit model to points
model.fit(X)
# Determine the cluster labels of new_points: labels
labels = model.predict([[4.6,3.4,1.8]])

x=df[df.target==0]
plt.scatter(x["sepal length (cm)"],x["petal length (cm)"],c="green")
y=df[df.target==1]
plt.scatter(y["sepal length (cm)"],y["petal length (cm)"],c="blue")
z=df[df.target==2]
plt.scatter(z["sepal length (cm)"],z["petal length (cm)"],c="yellow")
centroids = model.cluster_centers_
plt.scatter(centroids[:,0],centroids[:,2],marker=	'D',s=50,c="red")
plt.show()

x=df[df.target==0]
plt.scatter(x["sepal length (cm)"],x["sepal width (cm)"],c="green")
y=df[df.target==1]
plt.scatter(y["sepal length (cm)"],y["sepal width (cm)"],c="blue")
z=df[df.target==2]
plt.scatter(z["sepal length (cm)"],z["sepal width (cm)"],c="yellow")
centroids = model.cluster_centers_
plt.scatter(centroids[:,0],centroids[:,1],marker=	'D',s=50,c="red")
plt.show()
###############################################################################
# Load the library with the iris dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from sklearn.metrics import confusion_matrix

iris = load_iris()
# Create a dataframe with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target']=iris.target
df['species']=df['target'].replace({0:'setosa',1:'versicolor',2:'virginica'})

#Splitting to train and test
X_train, X_test, y_train, y_test = train_test_split(df[iris.feature_names], df['target'], random_state=1)
model=AgglomerativeClustering(n_clusters=3)
predict_val=model.fit_predict(X_train)

###############################################################################







###############################################################################
###NLP and Text Analytics

text="A line containing only whitespace, possibly with a comment, is known as a blank line and Python totally ignores it."
len(text)

text="A line containing only whitespace, possibly with a comment, is known as a blank line and Python totally ignores it."
tokens=text.split(' ')
print(len(tokens))
print((tokens))

new_tokens=[token for token in tokens if len(token)>=3]
print(new_tokens)

title_tokens=[token for token in tokens if token.istitle()]
print(title_tokens)

text=text.lower()


text="   A line containing only whitespace, possibly with a comment, is known as a blank line and Python totally ignores it.   "
tokens=text.split(' ')
print(tokens)

text="   A line containing only whitespace, possibly with a comment, is known as a blank line and Python totally ignores it.   "
text=text.strip()
tokens=text.split(' ')
print(tokens)

print(text.find('o'))
print(text.rfind('o'))
print(text.replace('o','O'))

###############################################################################
#Using regular expressions

import re
text="2018-02-11 03:33:08,358 [archive-weekly1_2428] INFO  [com.AbstractServiceCommand]: Calling command : "
date_pat='([0-9]{2,4}[-/][0-9]{2}[-/][0-9]{2})'
date_pat_comp=re.compile(date_pat)
date_pat_comp.findall(text)


import re
text="Oct 2017 [archive-weekly1_2428] INFO  [com.AbstractServiceCommand]: Calling command : "
date_pat='([Jan|Feb|Oct]+[ ][0-9]{2,4})'
date_pat_comp=re.compile(date_pat)
date_pat_comp.findall(text)

import pandas as pd

time_sentences = ["Monday: The doctor's appointment is at 2:45pm.", 
                  "Tuesday: The dentist's appointment is at 11:30 am.",
                  "Wednesday: At 7:00pm, there is a basketball game!",
                  "Thursday: Be back home by 11:15 pm at the latest.",
                  "Friday: Take the train at 08:10 am, arrive at 09:00am."]

df = pd.DataFrame(time_sentences, columns=['text'])
print(df)
print(df['text'].str.len())

time='[0-9]{1,2}[:][0-9]{1,2}'
df['text'].str.findall(time)

###############################################################################
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize,ne_chunk, pos_tag,Tree,ngrams
from nltk.stem import WordNetLemmatizer,PorterStemmer
#Reading Text

#http://www.nltk.org/book/
with open(r"D:\Work\DataScience\DS ppt\datascience notes\DS-8\DS-8\A Short History of Computers and Computing.txt","r") as inp_txt:
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

sent="John Works in ACTE and stays in Chennai"
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

inp = pd.read_excel(r"D:\Work\DataScience\DS ppt\datascience notes\DS-8\DS-8\Movie review.xlsx",encoding='utf-8' )
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