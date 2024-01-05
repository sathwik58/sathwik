import pandas as pd
import numpy as np

dataset = pd.read_csv('spam.csv', encoding='latin-1')
dataset = dataset[['v1', 'v2']]
dataset.rename(columns={'v1' : 'target' , 'v2' : 'text'} , inplace=True)

from sklearn.preprocessing import LabelEncoder 
ed = LabelEncoder()
dataset['target'] = ed.fit_transform(dataset['target'])   #spam is 1
dataset = dataset.drop_duplicates(keep='first')

# removing the punctu, stopwords, non alphanumeric, capsletters 
import nltk
from nltk.corpus import stopwords 
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text :
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation :
            y.append(i)
    text = y[:]
    y.clear()        
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)    
dataset['transform_text'] = dataset['text'].apply(transform_text)
print(dataset.columns)
#model 
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features = 3000)
x = tfidf.fit_transform(dataset['transform_text']).toarray()
y = dataset['target'].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,train_size=0.2,random_state=2)

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score

mnb = MultinomialNB()

mnb.fit(X_train,Y_train)
y_pred1 = mnb.predict(X_test)
print(accuracy_score(Y_test,y_pred1))
print(confusion_matrix(Y_test,y_pred1))
print(precision_score(Y_test,y_pred1))


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))

print("don")
