from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import argparse
import pickle as pkl
from flask import Flask, Response, request, render_template, jsonify
import base64


#********************YOLOv3 part********************
def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/messi.jpg")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    
    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img, classes, colors):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.25)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "160", type = str)
    return parser.parse_args()


#----------Setting YOLOv3----------
num_classes = 80

args = arg_parse()
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
  
CUDA = torch.cuda.is_available()
bbox_attrs = 5 + num_classes

model = Darknet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
    
assert inp_dim % 32 == 0 
assert inp_dim > 32

if CUDA:
    model.cuda()
            
model.eval()


def getFrame2(_img):
    img, orig_im, dim = prep_image(_img, inp_dim)        
    if CUDA:
       im_dim = im_dim.cuda()
       img = img.cuda()
            
            
    output = model(Variable(img), CUDA)
    output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
    
    
    output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim
    output[:,[1,3]] *= _img.shape[1]
    output[:,[2,4]] *= _img.shape[0]

            
    classes = load_classes('data/coco.names')
    colors = pkl.load(open("pallete", "rb"))
  

    list(map(lambda x: write(x, orig_im, classes, colors), output))
  
    ret, jpg = cv2.imencode("test.jpg", orig_im)
    return jpg.tostring()


#********************Flask part********************
app = Flask(__name__)


@app.route('/hello/')
def hello():
    return render_template('hello.html', title='YOLOv3Test')


@app.route("/img", methods=["POST"])
def img():
    #.....Base64→openCV.....
    enc_data  = request.form['img']
    dec_data = base64.b64decode(enc_data.split(',')[1])
    img_np = np.fromstring(dec_data, np.uint8)
    imgCV = cv2.imdecode(img_np, cv2.IMREAD_ANYCOLOR)

    #.....YOLOv3で処理.....
    imgYOLO = getFrame2(imgCV)

    #.....openCV→Base64.....
    dst_base64 = 'data:image/png;base64,'.encode('utf-8') + base64.b64encode(imgYOLO)
    
    #.....Make json for response.....
    ret = {
        'result' : dst_base64.decode('utf-8'),
    }
    return jsonify(ResultSet=ret)
    

app.run(ssl_context=('ssl/server.crt', 'ssl/server.key'), threaded=True, debug=True)