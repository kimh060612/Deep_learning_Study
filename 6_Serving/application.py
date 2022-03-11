import tensorflow as tf
from tensorflow import keras
from flask import Flask, render_template, request
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import json
import requests

app = Flask(__name__)

classes = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "boat", "truck"]
headers = {"content-type": "application/json"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    data = request.files["image"].read()
    image = Image.open(io.BytesIO(data))
    
    image = image.convert("RGB")
    image = image.resize((32, 32))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = np.asarray(image, dtype=np.uint8)
    
    data = json.dumps({"signature_name": "serving_default", "instances": image.tolist()})
    json_response = requests.post('http://localhost:8501/v1/models/ViT:predict', data=data, headers=headers)
    print(json_response.text)
    predictions = np.array(json.loads(json_response.text)["predictions"])
    index = np.argmax(predictions[0])
    
    return render_template("index.html", label=classes[index])
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)

