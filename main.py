from flask import request, Flask,jsonify
# import torch
# from PIL import Image
# import torchvision.transforms as transforms
# from torch.autograd import Variable
# from pytorch_model import CNN
# import json
# from io import BytesIO
# from base64 import b64decode

from flask_socketio import SocketIO
app = Flask(__name__)
cors = CORS(app, resources={r"https://oauth.ccxp.nthu.edu.tw//*": {"origins": "*"}})

@app.route("/test", methods=['GET'])
def test():
    print("test")
    return jsonify({'test': 'test'})