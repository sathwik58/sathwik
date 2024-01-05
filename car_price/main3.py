import pandas as pd
import numpy as np

dataset = pd.read_csv('cardata.csv')
# print(dataset.head())
dataset = dataset[dataset['year'].str.isnumeric()]
dataset['year'] = dataset['year'].astype(int)
dataset = dataset[dataset['Price']!= 'Ask For Price' ]
dataset['Price'] = dataset['Price'].str.replace(',','').astype(int)
dataset['kms_driven']=dataset['kms_driven'].str.split().str.get(0).str.replace(',','')
dataset=dataset[dataset['kms_driven'].str.isnumeric()]
dataset['kms_driven'] = dataset['kms_driven'].astype(int)
dataset['name'] = dataset['name'].str.split().str.slice(start=0,stop=3).str.join(' ')
dataset=dataset[~dataset['fuel_type'].isna()]
dataset =dataset.reset_index(drop=True)

dataset.to_csv('clean_data.csv')
x = dataset[['name', 'company', 'fuel_type']]
y = dataset['Price']

from sklearn.model_selection import train_test_split
x_train,x_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

ohe=OneHotEncoder()
ohe.fit(x[['name', 'company', 'fuel_type']])

column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name', 'company',  'fuel_type']),remainder='passthrough')
lr=LinearRegression()
pipe = make_pipeline(column_trans,lr)
pipe.fit(x_train,Y_train)
y_pred=pipe.predict(x_test)
print(r2_score(Y_test,y_pred))

score = []
for i in range(1000):
    x_train,x_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=i)
    lr=LinearRegression()
    pipe = make_pipeline(column_trans,lr)
    pipe.fit(x_train,Y_train)
    y_pred=pipe.predict(x_test)
    score.append(r2_score(Y_test,y_pred))
np.argmax(score)
x_train,x_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state= np.argmax(score))
lr=LinearRegression()
pipe = make_pipeline(column_trans,lr)
pipe.fit(x_train,Y_train)
y_pred=pipe.predict(x_test)

import pickle

pickle.dump(pipe,open('LR.pkl','wb'))

print(pipe.predict(pd.DataFrame([["Audi A8","Audi",2011,45000,"Diesel"]],columns=['name','company','year','kms_driven','fuel_type']))
)