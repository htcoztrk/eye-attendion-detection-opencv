# Import OpenCV2 for image processing
import cv2
#%%

vid_cam = cv2.VideoCapture(0)

#  Haarcascade Frontal Face ile yüz tanıma
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#  Haarcascade eye ile göz tanıma
eye = cv2.CascadeClassifier('haarcascade_eye.xml')
# tanımlanan her sınıf için id belirleme
class_id = 2

# Initialize sample face image
count = 0

# Start looping
while(True):

    # video frame
    _,image_frame = vid_cam.read()
    image_frame=cv2.flip(image_frame,1)
    # frame i grayscale e cevir
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    
    #yüzleri bul
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Loops for each faces
    for (x,y,w,h) in faces:

        # Crop the image frame into rectangle
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=image_frame[y:y+h,x:x+w]
        
        eyes=eye.detectMultiScale(roi_gray)
        i=0
        for(ex,ey,ew,eh) in eyes:
            i=i+1
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
            if(i==2):   count +=1   
            # image sayısını bir arttır
            

        
        #çekilen fotografları dataset sınıfına yazma
            cv2.imwrite("dataset/User." + str(class_id) + '.' + str(count) + ".jpg", roi_gray[ey:ey+eh,ex:ex+ew])

        
        #dikdortgen icine alınan yüzü göster
            cv2.imshow('frame', image_frame)

    # 'q' ya bas durdur
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # eger cekilen fotograf sayısı 100 'e ulaştıysa durdur.
    elif count>100:
        break

# Stop video
vid_cam.release()


cv2.destroyAllWindows()
