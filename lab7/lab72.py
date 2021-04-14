import cv2
import os
import numpy as np

onlyfiles = [f for f in os.listdir('wzorce') if os.path.isfile(os.path.join('wzorce', f))]
dmp = np.zeros((192, 64))
for f in onlyfiles:
    img = cv2.imread('wzorce/' + f, 0)
    _, img_bin = cv2.threshold(img, 40, 1, cv2.THRESH_BINARY)
    # cv2.imshow('asd', img_bin)
    # img_bin = cv2.medianBlur(img_bin, 3)
    # cv2.waitKey(1000)
    img_bin = cv2.resize(img_bin, (64, 192))
    dmp = dmp + img_bin
    cv2.imwrite('patterns_scaled/' + f, img_bin)
cv2.imwrite('main_pattern.png', dmp)
cv2.imshow('asd', dmp)
cv2.waitKey(0)