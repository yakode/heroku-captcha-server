from flask import request, Flask,jsonify
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from pytorch_model import CNN
import json
from io import BytesIO
from base64 import b64decode

app = Flask(__name__)


@app.route("/test", methods=['GET'])
def test():
    print("test")
    return jsonify({'test': 'test'})