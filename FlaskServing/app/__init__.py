#!/usr/bin/env python 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']  = "2"

import sys 
from os import path 

import json 
from datetime import datetime

from flask import Flask, Blueprint, redirect, request, url_for, jsonify, make_response

import tensorflow as tf
from tensorflow.python.keras import backend as k
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

#!----------------------------------------
# App Config
#!----------------------------------------
app = Flask(__name__, instance_relative_config=True)

# load config
app.config.from_object('config')
app.config.from_pyfile('application.cfg', silent=True)

# default controller 
@app.route('/', methods=['GET'])
def index():
    response = jsonify({
        'msg': 'Gender Prediction Api!',
        'path': request.path, 
        'systime': int(datetime.now().strftime('%s'))
    })
    response.status_code = 200
    return response

# load char embeddings
vocab_index = {}
with open('/'.join([app.config['BASEDIR'], "app/model_weight/char_dictionary.json"]), "r") as f:
    vocab_index = json.loads(f.read())

# load model here
gender_model = load_model('/'.join([app.config['BASEDIR'], "app/model_weight/gender_lstm_model.h5"]))



@app.route('/predict', methods=['POST'])
def predict():
    predicted_gender = 'UNK'

    # read input from users 
    payload = request.json 
    if 'name' in payload:
        name = payload['name']

        if len(name) > 32:
            response = jsonify({
                'msg': 'maximum character is 32',
                'path': request.path,
            })
            response.status_code = 412

        # predict with model 
        q_name = list(name.lower())
        test_dt = [vocab_index[x] for x in q_name]
        test_dt = pad_sequences([test_dt], maxlen=32)
        pad = np.array(test_dt[0])
        # predict with model
        res = gender_model.predict(pad.reshape(1, pad.shape[0]), batch_size=1, verbose=2)[0]

        conf_score = 0.0
        if np.argmax(res) == 0:
            predicted_gender = 'Female'
            conf_score = res[0] * 100
        elif np.argmax(res) == 1:
            predicted_gender = 'Male'
            conf_score = res[1] * 100
        
        response = jsonify({
            'msg': 'success predict gender',
            'name': name,
            'gender': predicted_gender,
            'confidence_score': '{:.2f}%'.format(conf_score)
        })
        response.status_code = 200

    else:
        response = jsonify({
            'msg': 'Name required',
            'path': request.path,
        })
        response.status_code = 200

    return response