import pandas as pd 
import numpy as np 
import json
import streamlit as st
import string
import random
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from nltk.corpus import stopwords
import nltk 
from nltk.stem import WordNetLemmatizer 
wl = WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer()
from sklearn.metrics.pairwise import cosine_similarity

data1 = json.load(open('intents.json'))
words =[]
classes = []
documents = []
ignor_letters = ['?' , '!' , ',' , '.']
for intent in data1['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [wl.lemmatize(word) for word in words if word not in  ignor_letters]
words = sorted(set(words))
classes =sorted(set(classes))

training = []
output_empty = [0]*len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [wl.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])  
random.shuffle(training)
training = np.arange(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model =SE

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text :
        if i.isalpha():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation  :
            y.append(i)
    text = y[:]
    y.clear()        
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)    
dataset = pd.read_csv('Recipe.csv')
dataset = dataset[['Title', 'Instructions','Cleaned_Ingredients']]
# print("hello")
# dataset['transform_ingredients'] = dataset['Cleaned_Ingredients'].apply(transform_text)
# dataset.to_csv('clean_data.csv')
print(dataset.columns)
dataset_clean = pd.read_csv('clean_data.csv')
dataset_clean.dropna(inplace=True)
# vector = cv.fit_transform(dataset_clean['transform_ingredients']).toarray()
# similarity = cosine_similarity(vector)

def opend(text):
    text = transform_text(text)
    total_text = [text] + dataset_clean['transform_ingredients']
    vector = cv.fit_transform(total_text)
    similarity = cosine_similarity(vector)
    similarity_score = similarity[0][1:]
    value = sorted(list(enumerate(similarity_score)) , reverse=True,key=lambda x:x[1])[1:11]
    for i in value:
        similar_recipe =dataset_clean.iloc[i[0]].Title
        print(similar_recipe)

print("enter the Ingredients")
text =input()
print(text)
opend(text)

