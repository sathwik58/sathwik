import numpy as np
import pandas as pd
movies_dataset = pd.read_csv('MovieDataset.csv')
movies_dataset = movies_dataset[['id' , 'original_title' , 'overview' , 'genres']]
movies_dataset['tags'] = movies_dataset['overview'] + movies_dataset['genres']
movies_dataset= movies_dataset.drop(columns=['overview' , 'genres' ])
movies_dataset['tags'] = movies_dataset['tags'].apply(lambda x:x.lower())


import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem(text):
    y =[]

    for i in text.split() :
        y.append(ps.stem(i))

    return " ".join(y)
movies_dataset['tags'] = movies_dataset['tags'].apply(stem)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=4000 , stop_words='english')
vectors =cv.fit_transform(movies_dataset['tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)


def recommend(movie) :
    movie_index = movies_dataset[movies_dataset['original_title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)) , reverse=True,key=lambda x:x[1])[1:11]

    for i in movie_list:
        print(movies_dataset.iloc[i[0]].original_title)

recommend('Ant-Man')

import pickle
pickle.dump(movies_dataset,open('movies.pkl','wb'))   
pickle.dump(similarity,open('similarity.pkl','wb'))