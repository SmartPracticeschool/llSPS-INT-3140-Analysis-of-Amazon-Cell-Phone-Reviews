from flask import Flask,request,render_template
from keras.models import load_model
import numpy as np
import pickle
global model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods =['POST'])
def predict():
    #Load the model
    import keras.backend.tensorflow_backend as tb
    tb._SYMBOLIC_SCOPE.value = True
    model = load_model('reviews.h5')
    entered_input = request.form['review']
    with open('CountVectorizer.pickle','rb') as file:
        cv= pickle.load(file)
        x_intent = cv.transform([entered_input])
        y_pred = model.predict(x_intent)
        print(y_pred)
    if (y_pred > 0.8):
        y = "Its a Positive Review"   
    else:
        y = "Its a Negative Review"

    return render_template('index.html', prediction_text = "Entered review: "+entered_input+"\nPrediction:" + y)


if __name__ == '__main__':
    app.run(debug = True)