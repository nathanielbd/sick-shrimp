import os
import numpy
#import Image

from flask import Flask, render_template, request
# from flask.ext.sqlalchemy import SQLAlchemy
# referenced from "Pretty Printed"
app = Flask(__name__)

@app.route("/")
def page():
	return render_template("index.html")
	
@app.route("/upload", methods = ["POST"])
def upload():
	file = request.files["inputFile"]
	from keras.models import Sequential
	from keras.layers import Conv2D, MaxPooling2D
	from keras.layers import Activation, Dropout, Flatten, Dense
	import os
	from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(100, 300, 3), data_format = 'channels_first'))
	model.add(Activation('relu'))
	
	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
				optimizer='rmsprop',
				metrics=['accuracy'])

	model.load_weights('try_one.h5')
	array = img_to_array(file)
	#result = model.predict(array, batch_size = 1, verbose = 0)
	return "model works"


if __name__ == "__main__":
	app.run(debug = True)


