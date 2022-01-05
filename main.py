import tflit
import numpy as np
from Pillow import Image
import requests
from io import BytesIO

from flask import request, Flask,jsonify
import json
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route("/captcha_pred", methods=['POST'])
def captcha_pred():
    req = json.loads(request.data.decode('utf-8'))
    result = main(req["url"], req["cookie"])
    return jsonify({'result': result})

@app.route("/test", methods=['POST'])
def captcha_pred():
    req = json.loads(request.data.decode('utf-8'))
    result = "1234"
    return jsonify({'result': result})

# A utility function to decode the output of the network
def decode(pred):
    
    # todo: fix further adjacent embedding predict the same number
    # eg. p = [11, 1, 11, 1, 11, 2, 3, 4] => ans = '11234' instead of '1234'
    p = np.argmax(pred, axis=2)
    to_num = ['?','4', '5', '0', '6', '3', '7', '9', '1', '8', '2']
    ans = ''
    for idx, i in enumerate(p[0]):
        if i == 11:
            continue
        # fix adjacent problem
        elif idx + 1 < len(p):
            if i == p[idx + 1]:
                continue
        else:
            ans += to_num[i]

    return ans

def load_image(url, PHPSESSID):
    response = requests.get(url, cookies={'PHPSESSID':PHPSESSID})
    img = Image.open(BytesIO(response.content)).convert('L')
    np_img = np.array(img)
    np_img = np_img / 255.
    np_img = np.expand_dims(np_img, axis=-1)
    np_img = np_img.transpose(1, 0, 2)
    np_img = np.expand_dims(np_img, axis=0)
    return np_img

def main(argv):
    model = tflit.Model('./model.tflite')    
    img = load_image(argv[1], argv[2])
    preds = model.predict(img)
    pred_texts = decode(preds)
    print(pred_texts)