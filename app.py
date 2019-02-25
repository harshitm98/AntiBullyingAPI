import flask
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from keras.models import model_from_json
import tensorflow as tf

app = flask.Flask(__name__)
global graph
graph = tf.get_default_graph()


def model(message):
    dataset = pd.read_csv("dataset/train.csv")
    X = dataset.iloc[:, 1].values
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
    cv.fit(X)
    with open("model/model.json", "r") as json_file:
        classifier = model_from_json(json_file.read())
    classifier.load_weights("model/model.h5")
    l = [[message]]
    test = np.array(l)
    test.astype('object')
    test = test.reshape((test.shape[0],))
    predicted = classifier.predict(cv.transform(test))
    predicted = int(predicted[0])
    if predicted == 0:
        return "Not toxic"
    elif predicted == 1:
        return "Toxic"
    else:
        return "Nibba calm down"


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    data = {"success": False}
    params = flask.request.json
    if params is None:
        params = flask.request.args

    if params is not None:
        with graph.as_default():
            data["response"] = model(params.get("msg"))
            data["success"] = True

    return flask.jsonify(data)


app.run(host="localhost")
