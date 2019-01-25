import numpy as np
import cv2
import pickle
import time

# Load in color image for face detection
image = cv2.imread('images/obamas4.jpg')

# Convert the image to RGB colorspace
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Make a copy of the original image to draw face detections on
image_copy = np.copy(image)

# Convert the image to gray 
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
# Detect faces in the image using pre-trained face dectector
faces = face_cascade.detectMultiScale(gray_image, 1.25, 6)

# Print number of faces found
print('Number of faces detected:', len(faces))

# Get the bounding box for each detected face
for f in faces:
    x, y, w, h = [ v for v in f ]
    cv2.rectangle(image_copy, (x,y), (x+w, y+h), (255,0,0), 3)
    # Define the region of interest in the image  
    face_crop = gray_image[y:y+h, x:x+w]
