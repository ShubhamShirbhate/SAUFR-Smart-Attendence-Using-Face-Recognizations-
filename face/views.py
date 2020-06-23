from django.shortcuts import render, redirect
import cv2, os
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
from project.settings import BASE_DIR
from .models import Student, Attendence
#project isername, ,SauFr password-admin
def first_page(request):
    return render(request, 'face/first_page.html')


def faculty(request):
    return render(request,'face/faculty.html')

def create_dataset(request):
    name =request.POST.get('name')
    Skey =request.POST.get('Skey')
    faceDetect = cv2.CascadeClassifier(BASE_DIR+'/saufr/haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)
    sampleNum = 0
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        for(x,y,w,h) in faces:
            sampleNum = sampleNum+1
            cv2.imwrite(BASE_DIR+'/saufr/dataset/'+str(Skey)+'.'+str(name)+'.'+str(sampleNum)+'.jpg', gray[y:y+h,x:x+w])
            cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2)
            cv2.waitKey(1)
        cv2.imshow("Face",img)
        cv2.waitKey(1)
        if(sampleNum>29):
            break
    cam.release()
    cv2.destroyAllWindows()
    student=Student(name=name, Skey=Skey)
    student.save()
    return redirect('/')




def trainer(request):
    import os
    from PIL import Image
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    #recognizer = cv2.face.LBPHFaceRecognizer_create()
    #recognizer =cv2.createLBPHFaceRecognizer()
    #Path of the samplesp
    path = BASE_DIR+'/saufr/dataset'
    def getImagesWithID(path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)] #concatinate the path with the image name
        faces = []
        Ids = []
        for imagePath in imagePaths:
            faceImg = Image.open(imagePath).convert('L') #convert it to grayscale
            faceNp = np.array(faceImg, 'uint8')
            ID = int(os.path.split(imagePath)[-1].split('.')[0])
            faces.append(faceNp)
            Ids.append(ID)
            cv2.imshow("training", faceNp)
            cv2.waitKey(100)
        return np.array(Ids), np.array(faces)
    ids, faces = getImagesWithID(path)
    recognizer.train(faces, ids)
    recognizer.save(BASE_DIR+'/saufr/recognizer/trainingData.yml')
    cv2.destroyAllWindows()
    return redirect('/')




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
            cv2.putText(img,Skey,(x,y+h), font,1,(255,255,255),3)
            aa = Attendence(Skey=Skey, date=date, time=timestamp)
        #tt=print(Skey+"-::-"+date+"-::-"+timestamp)
        cv2.imshow("img",img)
        if(cv2.waitKey(1) == ord('q')):
            break
    aa.save()  
    #print(tt)
    cam.release()
    cv2.destroyAllWindows()
    return redirect('/')
