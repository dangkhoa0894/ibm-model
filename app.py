import nltk
import json
import os
nltk.download('punkt')
from nltk_module.nltk_ibm import *
from models.ibm_model import *
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def app_check():
    print("Checking for dataset ...")
    print("Searching for en_fr.json")
    for f in os.listdir("./data"):
        if str(f).find("data1.json") or str(f).find("data2.json"):
            print("Dataset exists")
            return True
        else:
            print("Dataset doesn't exist")
            return False

def import_data(data=1):
    try:
        # IBM 1 and EM algorithm implementation
        if data == 1:
            with open('data/en_fr.json', encoding='utf-8') as f:
                data = json.loads(f.read())
                print(data)
                return data
        elif data == 2:
            # NLTK IBM 1 and IBM 2 and Phrase based extraction implementation.
            with open('data/data2.json', encoding='utf-8') as f:
                data = json.loads(f.read())
                return data
        else:
            with open('data/en_fr.json', encoding='utf-8') as f:
                data = json.loads(f.read())
                return data
    except Exception as e:
        print("Read error. Check file again")
        return False

@app.route('/api/translate', methods=['POST'])
def trans():
    try:
        json_ = request.json
        data = json_['data']
        option = json_['option']
        if option['model'] == 1: 
            result = nltk_ibm_one(data, int(option['iter']))
        if option['model'] == 2: 
            result = nltk_ibm_two(data, int(option['iter']))
        if option['model'] == 3: 
            result = nltk_ibm_three(data, int(option['iter']))
        return result.to_json()
    except Exception as e:
        resp = jsonify({'message': str(e)})
        resp.status_code = 400
        return resp

@app.route('/')
def index():
    return "<h1>DEMO NLTK</h1>"

if __name__=="__main__":
    app.run()
