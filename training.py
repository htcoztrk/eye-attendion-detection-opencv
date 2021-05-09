
# Import OpenCV2 for image processing
# Import os for file path
import cv2, os

# Import numpy 
import numpy as np

# Import Python Image Library (PIL)
from PIL import Image

recognizer=cv2.face.LBPHFaceRecognizer_create() 
# yüz tanıma için histogram oluşturma
#%%
# gözleri tanıma için haarcascade 
detector = cv2.CascadeClassifier("haarcascade_eye.xml");

# Create method to get the images and label data
#burada dataset klasorune kaydedilen fotograflar kullanılarak model oluşturuluyor
def getImagesAndLabels(path):

    #  file path
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    
    # 
    eyeSamples=[]
    
    # Initialize empty id
    ids = []

    # 
    for imagePath in imagePaths:

        # Get the image and convert it to grayscale
        PIL_img = Image.open(imagePath).convert('L')

        # PIL image i numpy ile array e cevir
        img_numpy = np.array(PIL_img,'uint8')

        #  image id i al
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        print(id)

        
        #egitilen görüntülerden gözleri al
        eyes = detector.detectMultiScale(img_numpy)

       
        for (x,y,w,h) in eyes:

            
            #görüntüleri eyeSamplesve ekle
            eyeSamples.append(img_numpy[y:y+h,x:x+w])

            # id yi ekle
            ids.append(id)

    
    return eyeSamples,ids


#göz ve id leri dataset klasorunden al
eyes,ids = getImagesAndLabels('dataset')

# gözleri ve id leri kullanarak modeli eğit
recognizer.train(eyes, np.array(ids))

#eğitilen modeli trainer.yml' ye kaydet
recognizer.save('trainer/trainer.yml')
