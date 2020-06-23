from django.shortcuts import render
from django.shortcuts import render, redirect
import cv2, os
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
from project.settings import BASE_DIR
from face.models import Student, Attendence
from face.views import detect

def home(request):
    return render(request,'faculty/home.html')

'''
def detect(request):
    faceDetect = cv2.CascadeClassifier(BASE_DIR+'/saufr/haarcascade_frontalface_default.xml')
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read(BASE_DIR+'/saufr/recognizer/trainingData.yml')
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x, y), (x+w,y+h), (255,0,0), 3)
            Skey,Conf = rec.predict(gray[y:y+h, x:x+w])
            if Conf >  50:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d/%m/%Y')
                timestamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                Skey = str(Skey)   
            else:
                Skey = 'Unknown'
                print(str(Skey))
            #tt=print(Skey+"-::-"+date+"-::-"+timestamp)
            cv2.putText(img,Skey,(x,y+h), font,1,(255,255,255),3)
      
       # print(tt)
        cv2.imshow("img",img)
        if(cv2.waitKey(1) == ord('q')):
            break

       
    cam.release()
    cv2.destroyAllWindows()
    aa = Attendence.all(Skey=Skey, date=date, time=timestamp)
    aa.save()  
   
    return redirect('/')

'''
