# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 21:04:04 2018

@author: Gurudeo
"""

#steps ar



import numpy as np
import cv2

#for fronatal face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#for eye
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#capture frame from video
cap = cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/trainningData.yml")
ID=0
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (0, 0, 255)
while 1:
    ret, img = cap.read()
    #convert  color to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #store frames in lists
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# Now perfrom operations on each frame stored in list and show on screen
    for (x,y,w,h) in faces:
        #creceate rectangle frame
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        ID,conf=rec.predict(gray[y:y+h,x:x+w])
        if(ID==1):
            ID= "Modi"
        elif(ID==2):
            ID= "Janvi kapoor"
        elif(ID==3):
            ID="Sachin Tendulkar"
        cv2.putText(img,str(ID),(x,y+h),fontface,fontscale,fontcolor)
        #Convert portion of frame for performing eye detection 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        #detects eye from 
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            #now color portion on eye
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    if(cv2.waitKey(1)==ord('q')):
         break;

cap.release()
cv2.destroyAllWindows()