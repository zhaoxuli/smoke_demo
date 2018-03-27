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




model_url = sys.argv[1]
epoch  = sys.argv[2]
input_size = 64
input_channel = 1
##init_model

Batch = namedtuple('Batch', ['data'])
sym , arg_params ,aux_params = mx.model.load_checkpoint(model_url,int(epoch))
mod = mx.mod.Module(symbol = sym ,context=mx.cpu() ,label_names =None)
mod.bind(for_training=False ,data_shapes = [('data',(1,input_channel,input_size,input_size))],label_shapes = mod._label_shapes)
mod.set_params(arg_params,aux_params,allow_missing =True)

def get_face_loc(img):
    loc = []
    cmd='~/alpha-det-prediction1.2.0/example/bin/AlphaDet_test  '+img
    print cmd
    output  = os.popen(cmd)
    messege = output.readlines()
    info = messege[4]
    if  info.split()[0] !='left':
        print messege
        return None,None
    else:
        loc.append(float(info.split()[1]))
        loc.append(float(info.split()[3]))
        loc.append(float(info.split()[5]))
        loc.append(float(info.split()[7]))
        conf = info.split()[9]
        return loc,conf

def get_roi(loc,img):
    src = cv2.imread(img,0)
    if src is None:
        print 'False  image'
    else :
        y1 = int(loc[1]+(loc[3]-loc[1])/2)
        #cv2.rectangle(src,(int(loc[0]),int(loc[1])),(int(loc[2]),int(loc[3])),(255,0,0),2)
        #cv2.rectangle(img,(int(rect[0]),int(rect[1])),(int(rect[2]),int(rect[3])),(255,0,0),2)
        half_face = src[int(y1):int(loc[3]),int(loc[0]):int(loc[2])]
        cv2.rectangle(src,(int(loc[0]),int(y1)),(int(loc[2]),int(loc[3])),(255,0,0),2)
        cv2.imwrite('./predict.png',half_face)
        return src

def Predict_Img(img_url):
    #process img
    src = cv2.imread(img_url , 0)
    img = cv2.resize(src,(64,64))
    img = img[np.newaxis, :]
    img = img[np.newaxis, :]
    mod.forward(Batch([mx.nd.array(img)]))
    #do predict
    prob = mod.get_outputs()[0].asnumpy()
    smoke_score = prob[0][0]
    return smoke_score

def main(video_path):
    videoCap = cv2.VideoCapture(video_path)
    frame_count = int(videoCap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    print frame_count
    count = 0
    while(int(count)<frame_count):
        count  = count +1
        videoCap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,count)
        ret,frame = videoCap.read()
        src = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imwrite('./temp.jpg',src)
        img_path = './temp.jpg'
        loc,conf = get_face_loc(img_path)
        if  loc is None:
            print 'non face'
            cv2.putText(src,'NONE_FACE',(10,30),cv2.FONT_HERSHEY_TRIPLEX,0.6,(255,255,255))
        else:
            src = get_roi(loc,img_path)
        #do predict
            smoke_score  = Predict_Img('./predict.png')
            print smoke_score
            #show the result
            cv2.putText(src,'smoke_score = '+str(smoke_score),(10,30),cv2.FONT_HERSHEY_TRIPLEX,0.6,(255,255,255))

            cv2.putText(src,'normal_score = '+str(1-float(smoke_score)),(10,70),cv2.FONT_HERSHEY_TRIPLEX,0.6,(255,255,255))
        cv2.imshow('main',src)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            count = frame_count
    cap.release()
    cv2.destroyWindows()


if  __name__ == '__main__':
    model_url = './mode'

    video_path = '/mnt/hgfs/Share/test/Smoke_demo.mp4'
    #img_path = '~/alpha-det-prediction1.2.0/example/123.png'
    main(video_path)
    #predict_Img
