from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
import cv2
import os

from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from werkzeug import secure_filename

# basedir = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def classify(img_face):
	custom_vgg_model = load_model('saved_model.h5')
	return custom_vgg_model.predict(img_face)

def predict(test_image):
	face_present = False
	modi_present = False
	kejriwal_present = False

	cl = ['arvind kejriwal', 'narendra modi']

	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

	img = cv2.imread(test_image)
	if img is None:
		return img, face_present, kejriwal_present, modi_present
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces_dec = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x, y, w, h) in faces_dec:
		face_present = True
		face = img[y:y+h, x:x+w]
		# if face.shape[0] < 160:
		face = cv2.resize(face, (224,224))
		im = np.zeros((1, 224, 224, 3))
		im[0,:,:,:] = face
		pred = classify(im)

		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
		# print(np.argmax(pred, axis=1))
		clno = np.argmax(pred, axis=1)[0]
		if clno == 0:
			kejriwal_present = True
		elif clno == 1:
			modi_present = True
		text = cl[np.argmax(pred, axis=1)[0]]
		cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

	return img, face_present, kejriwal_present, modi_present


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			return redirect(url_for('upload_done', filename=filename))
	return render_template('home.html')

@app.route('/<filename>', methods=['GET'])
def upload_done(filename):
	test_image = "static/upload/" + filename
	predicted_image, face_p, kejriwal_p, modi_p = predict(test_image)
	if predicted_image is not None:
		cv2.imwrite("static/outputs/"+filename, predicted_image)
	# return send_from_directory(app.config['UPLOAD_FOLDER'],
	#                            filename)
	print("static/outputs/"+filename)
	return render_template('display_result.html', dis_img="static/outputs/"+filename, face_p=face_p, kejriwal_p=kejriwal_p, modi_p=modi_p)

if __name__ == '__main__':
	app.run(debug=True)