from flask import Flask, request, render_template
import numpy as np
import pickle
import joblib

app = Flask(__name__, template_folder='templates')

@app.route('/') 
def web():
    return render_template('web.html')

@app.route('/predict', methods=["POST"])
def predict():
    petal_length = request.form['petal_length']
    sepal_length = request.form['sepal_length']
    petal_width = request.form['petal_width']
    sepal_width = request.form['sepal_width']

    sample_data = [sepal_length, sepal_width, petal_length, petal_width]
    clean_data = [float(i) for i in sample_data]

    ex1 = np.array(clean_data).reshape(1,-1)
    model=joblib.load('model/iris_model.pkl')
    result_prediction = model.predict(ex1)

    if result_prediction == 0:
        predicted_class = 'Iris-setosa'
    elif result_prediction == 1:
        predicted_class = 'Iris-versicolor'
    else:
        predicted_class = 'Iris-virginica'
        
    
    return render_template('web.html', prediction_text='Type of Iris: {}'.format(predicted_class)) 
   

if __name__=="__main__":
    app.run(port=5555, debug=True)