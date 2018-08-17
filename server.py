from keras.models import load_model
from PIL import Image

import tensorflow as tf
import numpy as np
import flask
import cv2
import io


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load():
	# load the pre-trained Keras model
	global model
	model = load_model('person.1.h5')
	global graph
	graph = tf.get_default_graph()

def prepare_image(image, target):
	# resize the input image and preprocess it
	image = cv2.resize(np.array(image), target)
	image = np.array( image).reshape(-1, 50,50, 1)
	image = image.astype('float32')
	image = image / 255.
	
	return image

@app.route("/predict", methods=["POST"])
def predict():
	
	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))
			image = prepare_image(image, target=(50, 50))

			with graph.as_default():
				predictions = model.predict(image)

			# return the predicted label
			predicted_classe = np.argmax(np.round(predictions),axis=1)

			labels_dict = np.load('labels_dict.npy').item()
			
			prediction = {}
			prediction['prediction'] = labels_dict.keys()[labels_dict.values().index(predicted_classe)]
			prediction['probability'] = float(max(predictions[0]))
			
	# return the results as a JSON response
	return flask.jsonify(prediction)

# starting the server
if __name__ == "__main__":
	load()
	app.run()