# import tensorflow as tf
# import tensorflow.keras.layers as layers
# import tensorflow.keras as keras
import tflite_runtime.interpreter as tflite
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

def decode(preds):
    np.argmax(preds, axis=2)
    p = np.argmax(preds, axis=2)
    to_num = ['?','4', '5', '0', '6', '3', '7', '9', '1', '8', '2']
    ans = ''
    for idx, i in enumerate(p[0]):
        if i == 11:
            continue
        # fix adjacent problem
        elif idx + 1 < len(p[0]) and i == p[0][idx + 1]:
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

def predict(img):
    interpreter = tflite.Interpreter(model_path="./model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #Structuring because TFlite doesn't do that
    input_shape = input_details[0]['shape']
    input_data = img.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    tflite_results = interpreter.get_tensor(output_details[0]['index'])

    del interpreter

    return tflite_results    

def main(_url, _cookie):
    img1 = load_image(_url, _cookie)
    img2 = load_image(_url, _cookie)
    img3 = load_image(_url, _cookie)
    pred1 = predict(img1)
    pred2 = predict(img2)
    pred3 = predict(img3)
    pred_text1 = decode(pred1)
    pred_text2 = decode(pred2)
    pred_text3 = decode(pred3)
    if pred_text1 == pred_text2:
        return pred_text1
    elif pred_text2 == pred_text3:
        return pred_text2
    else:
        return pred_text1

# # A utility function to decode the output of the network
# def decode_batch_predictions(pred):
    
#     # Construct look table by list, because I forgot to store it
#     characters = ['[UNK]', '4', '5', '0', '6', '3', '7', '9', '1', '8', '2']
#     num_to_char = layers.StringLookup(
#         vocabulary=characters, mask_token=None, invert=True
#     )
    
#     input_len = np.ones(pred.shape[0]) * pred.shape[1]
#     # Use greedy search. For complex tasks, you can use beam search
#     results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
#         :, :4 # max length = 4
#     ]
#     # Iterate over the results and get back the text
#     output_text = []
#     for res in results:
#         res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
#         output_text.append(res)
#     return output_text

# def load_image(url, PHPSESSID):
#     response = requests.get(url, cookies={'PHPSESSID':PHPSESSID})
#     img = Image.open(BytesIO(response.content)).convert('L')
#     img  = tf.keras.preprocessing.image.img_to_array(img)
#     img = tf.image.convert_image_dtype(img/255., tf.float32)
#     img = tf.image.resize(img, [80, 150])
#     img = tf.transpose(img, perm=[1, 0, 2])
#     img = tf.expand_dims(img, axis=0)
#     return img

# def main(_url, _cookie):
#     prediction_model = tf.keras.models.load_model('./prediction_model.h5', compile=False)
#     img = load_image(_url, _cookie)
#     preds = prediction_model.predict(img)
#     pred_texts = decode_batch_predictions(preds)
#     return pred_texts[0]
