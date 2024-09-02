import cv2 as cv
import numpy as np

img=cv.imread("istockphoto-464278425-1024x1024.jpg")
cv.imshow("lady",img)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("GREY",gray)

haar_cascade=cv.CascadeClassifier("haarcascade_frontalface_default.xml") # pre-trained library for face detection
# This file contains the trained data needed to detect faces.

fac_rect= haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
 # returns rectangular quardinates as a list of faces

# scaleFactor: how much the image is scaled down at each image scale

# minNeighbors: how many neighbors each rectangle should have to retain it as a face

# no. of faces is calculated by length
print(f'Number of faces found ={len(fac_rect)}')

# cascade censitive to the img

for(x,y,w,h) in fac_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
cv.imshow("Detected faces",img)
cv.waitKey(0)