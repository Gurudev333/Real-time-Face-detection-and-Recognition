# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 04:35:19 2018

@author: Gurudeo
"""
import os
import cv2
import numpy as np
import PIL
from PIL import Image
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def getImag_labels(path):
    Imgpaths=[os.path.join(path,f) for f in os.listdir(path)]
    Imgpaths.remove('DataSet\\Thumbs.db')
    faces=[]
    ids=[]
    for Imagepath in Imgpaths:
        #from each Imagepath convert image into gray scale image
        faceImage=Image.open(Imagepath)
        #images are working on numpy array so convert them to numpy arrayes
        faceNp=np.array(faceImage,'uint8')
        #Now to obtain from the image paths so split on slash and then split on dots
        ID=int(os.path.split(Imagepath)[-1].split('.')[1])

        faces.append(faceNp)
        ids.append(ID)
        cv2.waitKey(10)
        #now return faces and ids
    return faces,ids
        


faces,ids= getImag_labels('DataSet')
#train with faces and ids
recognizer.train(faces,np.array(ids))
#save the trainner
recognizer.save('recognizer/trainningData.yml')
cv2.destroyAllWindows()