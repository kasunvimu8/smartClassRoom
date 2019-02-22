import numpy as np
import cv2
import pickle
import time
import numpy as np
from tkinter import *
from tkinter import ttk
# from ttk import *
from os import walk
import cv2
from PIL import Image, ImageTk

head_count_g = 0

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

def logTranform(img) :   # more darker pixels map to light pixel in log transform
    max_ = np.max(img)
    log_transform_image =(255/np.log(1+max_)) * np.log(1+img)
   #cv2.imshow('log_transform_image',log_transform_image)
    return log_transform_image


def enhancingImage(gray):
    img=histogramEq(gray)
   # img=logTranform(img);
    # cv2.imshow('gray',gray)
    # cv2.imshow('gray',img)
    return img

def returnImage (path):

    frame =cv2.imread(path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processedImg = enhancingImage(gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
            #print(x,y,w,h)
            roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
            roi_color = frame[y:y+h, x:x+w]

            # recognize? deep learned model predict keras tensorflow pytorch scikit learn
            id_, conf = recognizer.predict(roi_gray)
            id_2, conf_eye = recognizer_eye.predict(roi_gray)
            # if i==1:
            #  print(conf , " " , conf_eye)
            if conf>=45 and conf <= 105 and conf_eye>=100 and conf_eye <= 150 :
                #print(5: #id_)
                #print(labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                name2 = eye_labels[id_2]
                color = (255, 255, 255)
                stroke = 2
                # if i==1:
                #   print(name," ",name2)
                if name == name2:
                    cv2.putText(frame, name, (x,y+20), font, 0.8, color, stroke, cv2.LINE_AA)

            color = (255, 0, 0) #BGR 0-255
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

            head_count_g = len(faces)

    print(" Number of Students : ", head_count_g) #number of persons
    return frame, len(faces)


image_directory = "testing_Image/"


# Get name of files in a directory
def get_image_names(file_name):
    f = []
    for (dirpath, dirnames, filenames) in walk(file_name):
        f.extend(filenames)
        break
    return f


def get_image(name):
    # Load an color image
    img, head_count_s = returnImage(name)
    img = cv2.resize(img, (600, 550))
    # cv2.imshow("image", img)

    # Rearrang the color channel
    b, g, r = cv2.split(img)
    img = cv2.merge((r, g, b))

    # Convert the Image object into a TkPhoto object
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im)

    return imgtk, head_count_s


def get_head_count():
    print(head_count_g)
    return head_count_g


root = Tk()
root.title("Smart Class room")

# Add a grid
mainframe = ttk.Frame(root)
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)
mainframe.pack(pady=10, padx=10)

# Create a Tkinter variable
tkvar = StringVar(root)
number_of_students = StringVar(root)

# Dictionary with options
choices = get_image_names(image_directory)
choices_set = set(choices)
tkvar.set(choices[0])  # set the default option

# Get image
imgtk,_ = get_image(image_directory+choices[0])
image_label = ttk.Label(mainframe, image=imgtk)
image_label.grid(row=0, column=1)

head_count_label = ttk.Label(mainframe, text="Number of peoples : "+str(_),  font=('Helvetica', 14))
head_count_label.grid(row=3, column=1)

popupMenu = ttk.OptionMenu(mainframe, tkvar, *choices)
Label(mainframe, text="Choose a Photo",).grid(row=1, column=1)
popupMenu.grid(row=2, column=1)


# on change dropdown value
def change_dropdown(*args):
    image_choice = tkvar.get()
    imgtk_s, head_count = get_image(image_directory + image_choice)
    image_label.config(image=imgtk_s)
    image_label.image = imgtk_s

    head_count_label.config(text="Number of peoples : "+str(head_count))


# link function to change dropdown
tkvar.trace('w', change_dropdown)

root.mainloop()


