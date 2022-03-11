import requests
import json
import cv2
import numpy as np

image = cv2.imread("./image/Shiba.jpg")
image = cv2.resize(image, dsize=(32, 32))
image = np.expand_dims(image, axis=0)

data = json.dumps({"signature_name": "serving_default", "instances": image.tolist()})

headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/ViT:predict', data=data, headers=headers)
predictions = np.array(json.loads(json_response.text)["predictions"])
print(np.argmax(predictions[0]))