import joblib
import numpy as np

def init():
    global model
    model = joblib.load('model.pkl')

def run(data):
    data = np.array(data['data'])
    result = model.predict(data)
    return result.tolist() 