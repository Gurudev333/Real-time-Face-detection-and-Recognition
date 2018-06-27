# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 02:31:53 2018

@author: Gurudeo
"""

import numpy as np
import cv2

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_detect = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
 
while 1:
    #read images from web cam
    ret,img=cap.read();
    #convert to grayscale
    gray=cv2.cvtColor(img,cv2.COLOR_BRG2GRAY);
    #store the images in list
    faces=face_detect.detectMultiScale(gray,1.3,5)
     for(x,y,w,h)

