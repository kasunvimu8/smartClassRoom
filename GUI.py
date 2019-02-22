from tkinter import *
import numpy as np
from tkinter import ttk
# from ttk import *
from os import walk
import cv2
from PIL import Image, ImageTk


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
    img = cv2.imread(name)
    img = cv2.resize(img, (500, 500))
    # cv2.imshow("image", img)

    # Rearrang the color channel
    b, g, r = cv2.split(img)
    img = cv2.merge((r, g, b))

    # Convert the Image object into a TkPhoto object
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im)

    return imgtk


def get_head_count(path):

    return 0


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
imgtk = get_image(image_directory+choices[0])
image_label = ttk.Label(mainframe, image=imgtk)
image_label.grid(row=0, column=1)

head_count_label = ttk.Label(mainframe, text="Number of peoples = 0",  font=('Helvetica', 14))
head_count_label.grid(row=3, column=1)

popupMenu = ttk.OptionMenu(mainframe, tkvar, *choices)
Label(mainframe, text="Choose a Photo",).grid(row=1, column=1)
popupMenu.grid(row=2, column=1)


# on change dropdown value
def change_dropdown(*args):
    image_choice = tkvar.get()
    imgtk_s = get_image(image_directory + image_choice)
    image_label.config(image=imgtk_s)
    image_label.image = imgtk_s

    head_count = get_head_count(image_directory + image_choice)
    head_count_label.config(text="Number of peoples = "+str(head_count))


# link function to change dropdown
tkvar.trace('w', change_dropdown)

root.mainloop()
