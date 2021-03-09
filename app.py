from flask import Flask, jsonify, request, Response
from flask_cors import CORS, cross_origin
from tensorflow import keras
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"
model = keras.models.load_model("cancer_model.h5")

@app.route("/malignant1", methods=["GET"])
@cross_origin()
def malignant1():
    aug = ImageDataGenerator(rescale=1 / 255.0)
    gen = aug.flow_from_directory(
        os.getcwd() + "/data/malignant1/",
        target_size=(48, 48),
        color_mode="rgb",
        shuffle=False,
        batch_size=1)

    predictions = model.predict(x=gen)
    classes = np.argmax(predictions, axis=1)

    _class = None
    if classes[0] == 1:
        _class = "malignant"
    else:
        _class = "benign"

    return jsonify({ "class": _class }), 200

@app.route("/malignant2", methods=["GET"])
@cross_origin()
def malignant2():
    aug = ImageDataGenerator(rescale=1 / 255.0)
    gen = aug.flow_from_directory(
        os.getcwd() + "/data/malignant2/",
        target_size=(48, 48),
        color_mode="rgb",
        shuffle=False,
        batch_size=1)

    predictions = model.predict(x=gen)
    classes = np.argmax(predictions, axis=1)

    _class = None
    if classes[0] == 1:
        _class = "malignant"
    else:
        _class = "benign"

    return jsonify({ "class": _class }), 200

@app.route("/malignant3", methods=["GET"])
@cross_origin()
def malignant3():
    aug = ImageDataGenerator(rescale=1 / 255.0)
    gen = aug.flow_from_directory(
        os.getcwd() + "/data/malignant3/",
        target_size=(48, 48),
        color_mode="rgb",
        shuffle=False,
        batch_size=1)

    predictions = model.predict(x=gen)
    classes = np.argmax(predictions, axis=1)

    _class = None
    if classes[0] == 1:
        _class = "malignant"
    else:
        _class = "benign"

    return jsonify({ "class": _class }), 200

@app.route("/benign1", methods=["GET"])
@cross_origin()
def benign1():
    aug = ImageDataGenerator(rescale=1 / 255.0)
    gen = aug.flow_from_directory(
        os.getcwd() + "/data/benign1/",
        target_size=(48, 48),
        color_mode="rgb",
        shuffle=False,
        batch_size=1)

    predictions = model.predict(x=gen)
    classes = np.argmax(predictions, axis=1)

    _class = None
    if classes[0] == 1:
        _class = "malignant"
    else:
        _class = "benign"

    return jsonify({ "class": _class }), 200

@app.route("/benign2", methods=["GET"])
@cross_origin()
def benign2():
    aug = ImageDataGenerator(rescale=1 / 255.0)
    gen = aug.flow_from_directory(
        os.getcwd() + "/data/benign2/",
        target_size=(48, 48),
        color_mode="rgb",
        shuffle=False,
        batch_size=1)

    predictions = model.predict(x=gen)
    classes = np.argmax(predictions, axis=1)

    _class = None
    if classes[0] == 1:
        _class = "malignant"
    else:
        _class = "benign"

    return jsonify({ "class": _class }), 200

@app.route("/benign3", methods=["GET"])
@cross_origin()
def benign3():
    aug = ImageDataGenerator(rescale=1 / 255.0)
    gen = aug.flow_from_directory(
        os.getcwd() + "/data/benign3/",
        target_size=(48, 48),
        color_mode="rgb",
        shuffle=False,
        batch_size=1)

    predictions = model.predict(x=gen)
    classes = np.argmax(predictions, axis=1)

    _class = None
    if classes[0] == 1:
        _class = "malignant"
    else:
        _class = "benign"

    return jsonify({ "class": _class }), 200

@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    image_file = request.files["image"]
    image_filename = image_file.filename
    image_file.save(os.path.join(os.getcwd() + "/tmp/tmp", image_filename))

    aug = ImageDataGenerator(rescale=1 / 255.0)
    gen = aug.flow_from_directory(
        os.getcwd() + "/tmp",
        target_size=(48, 48),
        color_mode="rgb",
        shuffle=False,
        batch_size=1)

    predictions = model.predict(x=gen)
    classes = np.argmax(predictions, axis=1)

    os.remove(os.getcwd() + "/tmp/tmp/" + image_filename)

    _class = None
    if classes[0] == 1:
        _class = "malignant"
    else:
        _class = "benign"

    return jsonify({ "class": _class }), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
