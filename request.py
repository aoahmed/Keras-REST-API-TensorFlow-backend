import requests

KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "faces_test/1-(9).pgm"

# load the input image
image = open(IMAGE_PATH, "rb").read()
# construct the payload for the request
payload = {"image": image}

# submit the request
req = requests.post(KERAS_REST_API_URL, files=payload).json()
print(req)
