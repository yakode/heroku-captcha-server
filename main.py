from flask import request, Flask,jsonify
# import torch
# from PIL import Image
# import torchvision.transforms as transforms
# from torch.autograd import Variable
# from pytorch_model import CNN
import json
# from io import BytesIO
# from base64 import b64decode

from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route("/test", methods=['POST'])
def test():
    res = json.loads(request.data.decode('utf-8'))
    return res