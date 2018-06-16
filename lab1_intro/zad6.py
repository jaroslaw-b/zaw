import cv2

import numpy as np

I = cv2.imread('lena.png', 0)
I_m = cv2.imread('mandril.jpg', 0)

I_r = I + I_m

cv2.imshow("C",np.uint8(I_r))
cv2.waitKey(0)

I_s = I - I_m
I_mn = I * I_m
I_comb = 2*I + 0.1*I_m
I_diff = cv2.absdiff(I, I_m)

cv2.imshow("odejmowanie", I_s)
cv2.imshow("mnożenie", I_mn)
cv2.imshow("kombinacja liniowa", np.uint8(I_comb))
cv2.imshow("absdiff", I_diff)
cv2.waitKey(0)