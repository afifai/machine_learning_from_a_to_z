import pickle
import pandas as pd
from flask import Flask, request
app = Flask(__name__)

with open("model/model_numpy.pkl", "rb") as model_file:
    model_numpy = pickle.load(model_file)

with open("model/model_pandas.pkl", "rb") as model_file:
    model_pandas = pickle.load(model_file)

FEATURES = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
LABEL = ['Iris Setosa', "Iris Versicolor", "Iris Virginica"]

@app.route('/')
def index():
    return {"status":"SUCCESS",
            "message":"Service is Up"}, 200

@app.route('/sapa')
def sapa_nama():
    args = request.args
    nama = args.get('nama', default="Afif")
    job = args.get('jobtitle', default="Data Scientist")
    return {"status":"SUCCESS",
            "message":f"Halo {nama}, pekerjaan anda adalah {job}"}, 200

@app.route('/predict/numpy')
def predict_numpy():
    args = request.args
    sl = args.get('sl', default=0.0, type=float)
    sw = args.get('sw', default=0.0, type=float)
    pl = args.get('pl', default=0.0, type=float)
    pw = args.get('pw', default=0.0, type=float)
    new_data = [[sl, sw, pl, pw]]
    res = model_numpy.predict(new_data)
    res = LABEL[res[0]]
    return {"status":"SUCCESS",
            "input type":"Numpy Array",
            "input":{
                'sepal length':sl,
                'sepal width':sw,
                'petal length':pl,
                'petal width': pw
                },
            "result":res}, 200

@app.route('/predict/pandas')
def predict_pandas():
    args = request.args
    sl = args.get('sl', default=0.0, type=float)
    sw = args.get('sw', default=0.0, type=float)
    pl = args.get('pl', default=0.0, type=float)
    pw = args.get('pw', default=0.0, type=float)
    new_data = [[sl, sw, pl, pw]]
    new_data = pd.DataFrame(new_data, columns=FEATURES)
    res = model_pandas.predict(new_data)
    res = LABEL[res[0]]
    return {"status":"SUCCESS",
            "input type":"Pandas DataFrame",
            "input":{
                'sepal length':sl,
                'sepal width':sw,
                'petal length':pl,
                'petal width': pw
                },
            "result":res}, 200

app.run(debug=True)