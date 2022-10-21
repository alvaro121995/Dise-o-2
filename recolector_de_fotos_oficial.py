#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:34:04 2022

@author: alvaroperaltaperalta
"""

import cv2
import os
import imutils

# Creación de carpeta cambiar variable name (Cambiar nombre del que se sacará fotos)
name = 'Alvaro'  #CAMBIAR SEGÚN LA PERSONA
path = '/Users/alvaroperaltaperalta/Desktop/Anaconda projects/Spyder anaconda/Recolector de datos Spyder/Data'
name_path = path + '/' + name
if not os.path.exists(name_path):
    print('Carpeta creada: ',name_path)
    os.makedirs(name_path)
    
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('Video.mp4')
faceClassif = cv2.CascadeClassifier('/Users/alvaroperaltaperalta/Desktop/Anaconda projects/Spyder anaconda/Recolector de datos Spyder/Data/haarcascade_frontalface_default.txt')
count = 0
while True:
    
    ret, frame = cap.read()
    if ret == False: break
    frame =  imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()
    faces = faceClassif.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(name_path + '/' + name + '_{}.jpg'.format(count),rostro)
        count = count + 1
    cv2.imshow('frame',frame)
    k =  cv2.waitKey(1)
    if k == 27 or count >= 300:
        break
cap.release()
cv2.destroyAllWindows()