import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
import numpy as np
from PIL import Image
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
def test():
    req = json.loads(request.data.decode('utf-8'))
    result = req["cookie"]
    return jsonify({'result': result})

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    
    # Construct look table by list, because I forgot to store it
    characters = ['[UNK]', '4', '5', '0', '6', '3', '7', '9', '1', '8', '2']
    num_to_char = layers.StringLookup(
        vocabulary=characters, mask_token=None, invert=True
    )
    
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :4 # max length = 4
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def load_image(url, PHPSESSID):
    response = requests.get(url, cookies={'PHPSESSID':PHPSESSID})
    img = Image.open(BytesIO(response.content)).convert('L')
    img  = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.image.convert_image_dtype(img/255., tf.float32)
    img = tf.image.resize(img, [80, 150])
    img = tf.transpose(img, perm=[1, 0, 2])
    img = tf.expand_dims(img, axis=0)
    return img

def main(_url, _cookie):
    prediction_model = tf.keras.models.load_model('./prediction_model.h5', compile=False)
    img = load_image(_url, _cookie[10:])
    preds = prediction_model.predict(img)
    pred_texts = decode_batch_predictions(preds)
    return pred_texts[0]