import cv2

camera = cv2.VideoCapture(0)
i = 1
while i <= 200:

    return_value, image = camera.read()
    cv2.imwrite('./images/Krishan/'+str(i)+'.jpg', image)
    i += 1
del(camera)
