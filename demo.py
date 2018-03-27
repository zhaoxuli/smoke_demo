# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 09:25:05 2017

@author: root
"""

import cv2
import logging
import numpy as np
import sys
import os
from time import clock
import mxnet as mx


print cv2.__version__
print cv2.__file__
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])


logger = logging.getLogger()
logger.setLevel(logging.INFO)

input_size = 72
input_channel = 1
prefix = "./model_chenzx/model"
num_round = 1000
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, num_round)
mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,input_channel,input_size,input_size))], label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)



def PreprocessGrayImage(src):
    img = src.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float")
    img = img - 128
    img = img * 0.01
    img = cv2.resize(img,(input_size, input_size))
    img = img[np.newaxis, :]
    img = img[np.newaxis, :]

    return img


def PreprocessColorImage(src):
    img = src.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float")
    img = img - 128
    img = img * 0.01
    img = cv2.resize(img,(input_size, input_size))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]

    return img

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    Rows are scores for each class.
    Columns are predictions (samples).
    """

    scoreMatExp = np.exp(np.asarray(x))
    return scoreMatExp / scoreMatExp.sum(0)


def predict(img):
    if input_channel == 1:
        input = PreprocessGrayImage(img)
    else:
        input = PreprocessColorImage(img)
    mod.forward(Batch([mx.nd.array(input)]))
    output = mod.get_outputs()[0].asnumpy()
    return np.squeeze(output)


def predict_single_image(img):
    #path = sys.argv[1]
    #img = cv2.imread(img_url)
    res = 90*predict(img)
    return res,img


def read_frame(cap, frameid):
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frameid)
    return cap.read()


if __name__ == '__main__':
    vd_path = '/mnt/hgfs/Share/modeltest_0721/haoming_sideface.mp4'
    cap  = cv2.VideoCapture(vd_path)
    if not cap.isOpened():
            print "ERROR : the mp4 file open failed."
            exit();
    frame_num = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    print 'total frame number is', frame_num

    cv2.namedWindow('main')
    for idx in xrange(frame_num):
        ret, img = read_frame(cap, idx)
        os.chdir('/home/chenzx/project/alpha_pre/alpha-det-prediction1.2.0/example/')

        cv2.imwrite("temp.jpg",img)
        start_face = clock()
        alpha ='./bin/AlphaDet_test temp.jpg'
        end_face = clock()
        print 'face comsuption:'+str(end_face-start_face)
        f = os.popen(alpha)
        data = f.readlines()
        #os.system(alpha)
        faceline = data[4].split()
        if faceline[0]=='left':
            rect=[]
            rect.append(float(faceline[1]))
            rect.append(float(faceline[3]))
            rect.append(float(faceline[5]))
            rect.append(float(faceline[7]))
            cv2.rectangle(img,(int(rect[0]),int(rect[1])),(int(rect[2]),int(rect[3])),(255,0,0),2)
            face = img[int(rect[1]):int(rect[3]),int(rect[0]):int(rect[2])]
            start_pose = clock()
            res,face = predict_single_image(face)
            end_pose = clock()
            print "pose comsumption:"+str(end_pose-start_pose)
            #print res
            cv2.putText(img,"pitch = " + str(res[0]), (10,30),cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0,0,255))
            cv2.putText(img,"yaw = " + str(res[1]), (10,50),cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0,0,255))
            cv2.putText(img,"roll = " + str(res[2]), (10,70),cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0,0,255))
            cv2.imshow("main", img)
            #cv2.waitKey(0)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        else:
            cv2.putText(img,"non face",(10,40),cv2.FONT_HERSHEY_TRIPLEX,1,(0,255,0))
            cv2.imshow('main',img)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
