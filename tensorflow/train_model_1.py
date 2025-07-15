#!/usr/bin/env python3

import cv2 as cv
#import tensorflow as tf
import time
import uuid
import os

labels = ["thumbsup", "thumbsdown", "thankyou", "livelong"]
number_imgs = 5

img_path = os.path.join('ml_images', 'collected_images')


if not os.path.exists(img_path):
    if os.name == 'posix':
        os.makedirs(img_path)

for label in labels:
    path = os.path.join(img_path, label)
    if not os.path.exists(path):
        os.makedirs(path)


cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print('failed to capture image')
        break

    flipped = cv.flip(frame, 1)
    cv.imshow("frame", flipped)
    for label in labels:
        print("Collecting {} images".format(label))

        for index in range(number_imgs):  
            #imgname = os.path.join(img_path, label, label + '_{}.jpg'.format(str(index)))
            cv.imshow('frame', flipped)
            #cv.imwrite(imgname, flipped)
            print("Collected image {}".format(index))
            time.sleep(0.5)

    print("Image collection done!!")
    if cv.waitKey(20) and 0xff == ord('d'):
        break


cap.release()
cv.destroyAllWindows()
