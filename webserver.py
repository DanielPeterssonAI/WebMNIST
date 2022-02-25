from tensorflow import keras
import numpy as np
from flask import Flask
from flask import request
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

@app.route('/')
def hello():
    with open("index.html") as f:
        lines = f.readlines()

    return ''.join(lines)

@app.route('/predict', methods = ["GET", "POST"])
def predict():
    data = str(request.get_data())[2:-2].split(",")
    data = list(map(int, data))
    data = np.array(data).reshape(1, -1)
    data = 255 - data
    
    y_pred = model_KNN.predict(data)

    return f"KNN predicted: {y_pred[0]}"

def knn_predictor(x_predict, model):
    return model.predict(x_predict)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

model_KNN = KNeighborsClassifier(n_neighbors = 1)
model_KNN.fit(x_train, y_train)

if __name__ == '__main__':
    app.run()