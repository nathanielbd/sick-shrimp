import os
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
	file = request.form["inputFile"]
	#image = Image.open(file)
	#image.show()

	return file.filename

if __name__ == "__main__":
	app.run(debug = True)


