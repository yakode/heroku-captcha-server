# import tensorflow as tf
# import tensorflow.keras.layers as layers
# import tensorflow.keras as keras
from concurrent.futures import ThreadPoolExecutor
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import threading

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

def load_image(url_and_cookie):
    url, cookie = url_and_cookie
    response = requests.get(url, cookies={'PHPSESSID':cookie})
    img = Image.open(BytesIO(response.content)).convert('L')
    np_img = np.array(img)
    np_img = (np_img / 255.).astype(np.float32)
    np_img = np.expand_dims(np_img, axis=-1)
    np_img = np_img.transpose(1, 0, 2)
    np_img = np.expand_dims(np_img, axis=0)
    return np_img

def load_images(url, PHPSESSID):
    url_and_cookie = [(url, PHPSESSID)] * 5
    pool = ThreadPoolExecutor(max_workers=5)

    images = []

    for image in pool.map(load_image, url_and_cookie):
        images.append(image)

    return images
        
def predict(images):
    interpreter = tflite.Interpreter(model_path="./model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #Structuring because TFlite doesn't do that
    tflite_results = []
    for i in images:
        interpreter.set_tensor(input_details[0]['index'], i)

        interpreter.invoke()

        tflite_result = interpreter.get_tensor(output_details[0]['index'])
        tflite_results.append(decode(tflite_result))

    del interpreter
    
    return max(tflite_results,key=tflite_results.count) 

def main(url, PHPSESSID):
    img_ = load_images(url, PHPSESSID)
    pred = predict(img_)
    return pred
# def decode(preds):
#     np.argmax(preds, axis=2)
#     p = np.argmax(preds, axis=2)
#     to_num = ['?','4', '5', '0', '6', '3', '7', '9', '1', '8', '2']
#     ans = ''
#     for idx, i in enumerate(p[0]):
#         if i == 11:
#             continue
#         # fix adjacent problem
#         elif idx + 1 < len(p[0]) and i == p[0][idx + 1]:
#                 continue
#         else:
#             ans += to_num[i]

#     return ans

# def load_image(url, PHPSESSID):
#     response = requests.get(url, cookies={'PHPSESSID':PHPSESSID})
#     img = Image.open(BytesIO(response.content)).convert('L')
#     np_img = np.array(img)
#     np_img = np_img / 255.
#     np_img = np.expand_dims(np_img, axis=-1)
#     np_img = np_img.transpose(1, 0, 2)
#     np_img = np.expand_dims(np_img, axis=0)
#     return np_img

# def predict(img):
#     interpreter = tflite.Interpreter(model_path="./model.tflite")
#     interpreter.allocate_tensors()

#     # Get input and output tensors
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     #Structuring because TFlite doesn't do that
#     input_shape = input_details[0]['shape']
#     input_data = img.astype(np.float32)
#     interpreter.set_tensor(input_details[0]['index'], input_data)

#     interpreter.invoke()

#     tflite_results = interpreter.get_tensor(output_details[0]['index'])

#     del interpreter

#     return tflite_results    

# def main(url, PHPSESSID):
#     img_ = load_image(url, PHPSESSID)
#     pred = predict(img_)
#     pred_text = decode(pred)
#     return pred_text
# results = [none] * 3
# threads = [none] * 3
# def laigao(url, PHPSESSID, result, index):
#     img_ = load_image(url, PHPSESSID)
#     pred = predict(img_)
#     pred_text = decode(pred)
#     result[index] = pred_text


# def main(_url, _cookie):
#     for i in range(len(threads)):
#         threads[i] = Thread(target=laigao, args=(_url, _cookie, results, i))
#         threads[i].start()
#     for i in range(len(threads)):
#         threads[i].join()

#     if results[0] == results[1]:
#         return results[0]
#     elif results[1] == results[2]:
#         return results[1]
#     else:
#         return results[0]

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
