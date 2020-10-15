from flask import Flask, jsonify, render_template, request
import numpy as np;
import pickle

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def root():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    
    # pred=model.predict([sf])
    # return render_template('result.html', prediction_text='The review is "{}"'.format(pred[0]))
     return render_template('index.html', prediction_text='heelo')

    

    
if __name__ == '__main__':
    app.run(debug=True)