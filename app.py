from distutils.log import debug
from flask import Flask,request,jsonify,render_template,redirect,url_for
import sklearn
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor

app=Flask(__name__)
models=pickle.load(open('ExamenLabo.pkl',"rb"))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    models=pickle.load(open('ExamenLabo.pkl',"rb"))
    int_futures=[float(i) for i in request.form.values()]
    dernier_futures=[np.array(int_futures)]
    dernier_futures=np.array([dernier_futures]).reshape(1,8)
    predire=models.predict(dernier_futures)
    if(models.predict(dernier_futures)==0):
        predire="Negatif"
    else:
        predire="Positif"
    return render_template('index.html', prediction_text_="Votre Test est : {}".format(predire))
if __name__=='__main__':
    app.run(debug=True)