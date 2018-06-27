# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 02:59:45 2018

@author: Gurudeo
"""

import numpy as np
import cv2
import sqlite3
#for fronatal face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#for eye
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#capture frame from video
cap = cv2.VideoCapture(0)
def insertupdate(Id,Name):
    conn=sqlite3.connect('face_rec.db')
    cmd="SELECT * FROM person WHERE ID="+str(Id)
    cursor=conn.execute(cmd)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):
        cmd="UPDATE Person SET Name="+str(Name)+"WHERE ID="+str(Id)
    else:
        cmd="INSERT INTO Person(ID,Name) Values("+str(Id)+","+str(Name)+")"
    conn.execute(cmd)
    conn.commit()
    conn.close()
id=input('enter user id')
name=input('enter name')

insertupdate(id,name)
id_no=0
while 1:
    ret, img = cap.read()
    #convert  color to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #store frames in lists
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# Now perfrom operations on each frame stored in list and show on screen
    for (x,y,w,h) in faces:
        #before perforrming operation on frae write on 
        id_no=id_no+1
        cv2.imwrite("DataSet/person."+str(id)+"."+str(id_no)+".jpg",gray[y:y+h,x:x+w])
        #creceate rectangle frame
        
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #Convert portion of frame for perform1ing eye detection 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        #detects eye from 
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            #now color portion on eye
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.waitKey(100)
    cv2.imshow('img',img)
    cv2.waitKey(1)
    if(id_no>20):
         break;

cap.release()
cv2.destroyAllWindows()