import numpy as np
import cv2
import pickle
import time
import numpy as np

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

#faces
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

#eyes
recognizer_eye = cv2.face.LBPHFaceRecognizer_create()
recognizer_eye.read("./recognizers/eye-trainner.yml")

#faces
labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

#eyes
eye_labels = {"person_name": 1}
with open("pickles/eye-labels.pickle", 'rb') as f:
    eye_og_labels = pickle.load(f)
    eye_labels = {v:k for k,v in eye_og_labels.items()}

cap = cv2.VideoCapture(0)

def histogramEq(img) :
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
   cl1 = clahe.apply(img)
   return cl1

def logTranform(img) :
    max_ = np.max(gray)
    log_transform_image =(255/np.log(1+max_)) * np.log(1+gray)
  #cv2.imshow('log_transform_image',log_transform_image)
    return log_transform_image


def enhancingImage(gray):
    img=histogramEq(gray)
    img=logTranform(img);
    # cv2.imshow('gray',gray)
    # cv2.imshow('gray',img)
    return img


while(True):
    time.sleep(0.1)
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processedImg=enhancingImage(gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
        roi_color = frame[y:y+h, x:x+w]

        # recognize? deep learned model predict keras tensorflow pytorch scikit learn
        id_, conf = recognizer.predict(roi_gray)
        id_2, conf_eye = recognizer_eye.predict(roi_gray)
      #   print(conf , " " , conf_eye)
        if conf>=40 and conf <= 85 and conf_eye>=100 and conf_eye <= 150:
            #print(5: #id_)
            #print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            name2 = eye_labels[id_2]
            color = (255, 255, 255)
            stroke = 2
          #  print(name," ",name2)
            if name == name2:
                cv2.putText(frame, name, (x,y-15), font, 0.8, color, stroke, cv2.LINE_AA)


        color = (255, 0, 0) #BGR 0-255 
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        #subitems = smile_cascade.detectMultiScale(roi_gray)
        #for (ex,ey,ew,eh) in subitems:
        #   cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


    print(" Number of Students : ", len(faces)) #number of persons

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


