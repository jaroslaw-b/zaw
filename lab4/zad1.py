import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.signal import argrelextrema
import scipy.ndimage.filters as filters

filelist = os.listdir('img/')

def harris(img, size):
    gauss_size = (size, size)
    img_x = cv2.Sobel(img , cv2.CV_32F, 1, 0, ksize=size)
    img_y = cv2.Sobel(img , cv2.CV_32F, 0, 1, ksize=size)
    Ixx = img_x*img_x
    Iyy = img_y*img_y
    Ixy = img_x*img_y
    Ixx = cv2.GaussianBlur(Ixx, gauss_size, 0)
    Iyy = cv2.GaussianBlur(Iyy, gauss_size, 0)
    Ixy = cv2.GaussianBlur(Ixy, gauss_size, 0)
    det_m = Ixx * Iyy - Ixy * Ixy
    tr_m = Ixx + Iyy
    h = det_m - 0.05 * tr_m*tr_m
    threshold = np.amax(h)*0.1
    # print(h)
    # cv2.adaptiveThreshold(h, 255, cv2.)
    data_max = filters.maximum_filter(h, 5)
    maxima = (h == data_max)
    data_min = filters.minimum_filter(h, 5)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    image = np.zeros(h.shape)
    image[maxima] = 1
    maxima = np.argwhere(image)
    return maxima

def gauss(img, sigma, k):
    images = []
    for i in range(k):
        images.append(cv2.GaussianBlur(img, ksize=0, sigmaX=sigma*(k+1), sigmaY=sigma*(k+1)))

for i in filelist:
    img_clr = cv2.imread('img/'+i)
    img = cv2.cvtColor(img_clr, cv2.COLOR_RGB2GRAY)
    maxima = harris(img, 3)
    for m in maxima:
        cv2.circle(img_clr, (m[1], m[0]), 2, (0,0,255), -1)
    cv2.imshow("asd", img_clr)
    cv2.waitKey(0)
