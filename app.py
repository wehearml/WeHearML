# import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask import jsonify
from joblib import load
import json

# Initialize the flask App
app = Flask(__name__)

# Mapping Dictionary

tab_map = {
    "neurobin_vit_b_pink_red": 0,
    "dolo_650": 1,
    "ecosprin": 2,
    "lemolate_gold": 3,
    "danp": 4,
    "lth_unison": 5
}

milk_map = {
    'shakti_6ml': 0,
    'shakti_7ml': 1,
    'shakti_8ml': 2,
    'shakti_9ml': 3,
    'shakti_10ml': 4,
    'gold_6ml': 5,
    'gold_7ml': 6,
    'gold_8ml': 7,
    'gold_9ml': 8,
    'gold_10ml': 9,
    'taza_10ml': 10,
    'taza_8ml': 11,
    'taza_9ml': 12,
    'taza_7ml': 13,
    'taza_6ml': 14,
    'taza_5ml': 15,
    'shakti_5ml':16,
    'gold_5ml':17
}

tab_op_map = {value: key for key, value in tab_map.items()}
milk_op_map = {value: key for key, value in milk_map.items()}

# model = pickle.load(open('final_model.pkl', 'rb'))
tab_model = load('tab_model.pkl')
milk_model = load('milk_model.pkl')


# default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')


# milk

@app.route('/predict-milk', methods=['GET', 'POST'])
def prediction_milk_api():
    if request.method == 'GET':
        arr = json.loads(request.args.get('arr', None))
    else:
        arr = request.get_json()
    arr = np.array(arr['arr'])
    preds = milk_model.predict_proba(arr.reshape(1, -1))
    acc = np.amax(preds)
    output = milk_op_map.get(preds.argmax())
    return jsonify(milk_name=output, acc=str(acc * 100))


# tab

@app.route('/predict-tab', methods=['GET', 'POST'])
def prediction_tab_api():
    if request.method == 'GET':
        arr = json.loads(request.args.get('arr', None))
    else:
        arr = request.get_json()
    arr = np.array(arr['arr'])
    preds = tab_model.predict_proba(arr.reshape(1, -1))
    acc = np.amax(preds)
    output = tab_op_map.get(preds.argmax())
    return jsonify(med_name=output, acc=str(acc * 100))


if __name__ == "__main__":
    app.run(debug=True)
