import pickle
import streamlit as st
import requests

movies =pickle.load(open('movies.pkl' , 'rb'))
similarity =pickle.load(open('similarity.pkl' , 'rb'))
movie_list = movies['original_title'].values

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=c7ec19ffdd3279641fb606d19ceb9bb1&language=en-US".format(movie_id)
    data=requests.get(url)
    data=data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/"+poster_path
    return full_path


def recommend(movie) :
    movie_index = movies[movies['original_title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)) , reverse=True,key=lambda x:x[1])[1:11]
    recommend_movie = []

    for i in movie_list:
        movies_id=movies.iloc[i[0]].id
        recommend_movie.append(movies.iloc[i[0]].original_title)
    return recommend_movie


st.header('Movie Recommender System')
movie_selected = st.selectbox("Select the movie",movie_list)
if st.button("Show Recommend"):
    movies_names=recommend(movie_selected)
    row1,col2,col3,col4,col5 =st.columns(5)
    with row1:
        st.text(movies_names[0])
    with col2:
        st.text(movies_names[1])
    with col3:
        st.text(movies_names[2])
    with col4:
        st.text(movies_names[3])
    with col5:
        st.text(movies_names[4])       
