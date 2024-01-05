from flask import Flask , render_template ,request
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np


app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('LR.pkl','rb'))
cars = pd.read_csv('clean_data.csv')
@app.route('/')
def index():
    companies = sorted(cars['company'].unique())
    model =sorted( cars['name'].unique())
    year = sorted(cars['year'].unique(), reverse=True)
    fule = sorted(cars['fuel_type'].unique())
    return render_template("web3.html",companies=companies , model=model ,year=year,fule=fule)

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    company= request.form.get('companies')

    car_model=request.form.get('model')
    year=request.form.get('year')
    fuel_type=request.form.get('fule')
    driven=request.form.get('kilo_driven')

    prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'], data=np.array([car_model,company,year,driven,fuel_type]).reshape(1, 5)))
    print(prediction)

    return str(np.round(prediction[0],2))



if __name__=='__main__':
    app.run()