from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template
import flask
import os
import sys
import numpy as np
import tensorflow as tf
import pickle
import pdb
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from flask import Flask
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

app = flask.Flask(__name__)

def get_model():
	global model
	model = load_model('emoji_model.h5')
	print('Model Loaded!!')


tf.compat.v1.get_default_graph()

tokenizer = pickle.load(open('tokenizer.pickle','rb'))

@app.route('/')
def home():
	return render_template('view.html')

@app.route('/predict',methods = ['GET''POST'])
def predict():
	global graph
	global tokenizer
	with graph.as_default():
		maxlen = 50
		text = request.form['name']
		test_sent = tokenizer.texts_to_sequences([text])
		test_sent = pad_sequences(test_sent, maxlen = maxlen)
		pred = model.predict(test_sent)
		response = {
		'prediction': int(np.argmax(pred))
		}
	return flask.jsonify(response)


@app.route('/update',methods = ['GET"POST'])
def update():
	global graph
	global tokenizer
	with graph.as_default():
		maxlen = 50
		text = request.form['sentence']
		test_sent = tokenizer.texts_to_sequences([text])
		test_sent = pad_sequences(test_sent, maxlen = maxlen)
		test_sent = np.vstack([test_sent] * 5)
		actual_output = request.form['dropdown_value']
		output_hash = {
			'Happy': np.array([1.,0.,0.,0.,0.,0.,0.]),
			'Fear': np.array([0.,1.,0.,0.,0.,0.,0.]),
			'Anger': np.array([0.,0.,1.,0.,0.,0.,0.]),
			'Sadness': np.array([0.,0.,0.,1.,0.,0.,0.]),
			'Disgust': np.array([0.,0.,0.,0.,1.,0.,0.]),
			'Shame': np.array([0.,0.,0.,0.,0.,1.,0.]),
			'Guilt': np.array([0.,0.,0.,0.,0.,0.,1.]),
					}
		actual_output = output_hash[actual_output].reshape((1,7))
		actual_output = np.vstack([actual_output] * 5)
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.fit(test_sent, actual_output, epochs = 10, batch_size = 32, shuffle=True)
		model.save('emoji_model.h5')
		get_model()
		response = {
		'update_text': 'Updated the values!! Should work in next few attempts..'
		}
	return flask.jsonify(response)

if __name__ == '__main__':
	app.run(threaded=True)