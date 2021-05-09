# Import OpenCV2 for image processing
import cv2

# Import numpy 
import numpy as np
#%%
#yüz recognition için local pattern oluştur.
recognizer=cv2.face.LBPHFaceRecognizer_create() 



#trainer dosyasında kameradan resimler çekilerek moluşturulan modeli okuma işlemi
recognizer.read("trainer/trainer.yml")

#yüz tanıma için modeli load etme
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
eye = cv2.CascadeClassifier("haarcascade_eye.xml");
# font style yi ayarlama
font = cv2.FONT_HERSHEY_SIMPLEX

#videocapture yi initialize etme
cam = cv2.VideoCapture(0)

# Loop
while True:
    # video framei okuma
    ret, im =cam.read()
    im=cv2.flip(im,1)
    # frame i grayscale e çevirme
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # frame deki tüm yüzleri alma
    faces = faceCascade.detectMultiScale(gray, 1.2,5)

    # For each face in faces
    for(x,y,w,h) in faces:
      
        # yüz etrafinda dikdortgen olusturma
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=im[y:y+h,x:x+w]
        
        eyes=eye.detectMultiScale(roi_gray)
        # yüzdeki gözleri tanıma
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
            Id,conf= recognizer.predict(roi_gray[ey:ey+eh,ex:ex+ew])
            # recognize edilen ID nin hangi sınıfa ait oldugunu belirleme
            if(Id==1):  Id="dikkatli"
            elif(Id==2):    Id="dikkatsiz"
            else:   Id="bilemedim"
            cv2.rectangle(im, (x-22,y-60), (x+w+22, y-22), (0,255,0), -1)
            cv2.putText(im, str(Id), (x,y-20), font, 1, (255,255,255), 2)
        #cv2.rectangle(im, (x-22,y-60), (x+w+22, y-22), (0,255,0), -1)

    # videoframe ve diktortgeni gösterir
    cv2.imshow('im',im) 

    # "q" ya bas programı kapat
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# camerayı durdurma
cam.release()

# tüm pencereli kapat
cv2.destroyAllWindows()
