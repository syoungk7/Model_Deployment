import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

# apply flask
app=Flask(__name__)

# home
@app.route('/')

# index
@app.route('/index')
def index():
    return flask.render_template('index.html')

def ValuePredictor(to_predict_list): # for prediction
    to_predict = np.array(to_predict_list).reshape(1, 12)
    loaded_model = pickle.load(open("model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

# result
@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)

        if int(result) == 1:
            prediction = 'Your predicted income is more than 50K'
        else:
            prediction = 'Your predicted income is less than 50K'

        return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
	app.run(debug=True)