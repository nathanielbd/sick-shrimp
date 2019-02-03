import os
import numpy as np
import cv2

#import Image

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
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
    
	# path = '/UPLOAD_FOLDER'
	# os.mkdir(path)
	# app.config['UPLOAD_FOLDER'] = path

	path = '/UPLOAD_FOLDER'
	app.config['UPLOAD_FOLDER'] = path

	filename = secure_filename(file.filename) # save file 
	filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
	file.save(filepath)

	img = cv2.imread(filepath)
	img = cv2.resize(img, (100, 300))
	print(img)
	img = numpy.reshape(img, [1, 100, 300, 3])
	classes = model.predict_classes(img)
	# os.chown(path, uid, gid)

	#file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

	#img = load_img('./'+file.filename, target_size=(100, 300))
	
	#array = img_to_array(file)
	#result = model.predict(array, batch_size = 1, verbose = 0)
	return classes


if __name__ == "__main__":
	app.run(debug = True)


