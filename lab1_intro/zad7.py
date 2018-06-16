import numpy as np
import cv2
from matplotlib import pyplot as plt
def hist(img):
    h=np.zeros((256,1), np.float32) # tworzy i zeruje tablice
    height, width = img.shape[:2]  # shape - krotka z wymiarami - bierzemy     2     pierwsze
    for y in range(height):
        for x in range(width):
            h[img[x, y]] += 1
    return h

I = cv2.imread('lena.png', 0)

hist = hist(I)
plt.figure(1)
plt.plot(hist, 'r.')
plt.show()


IGE = cv2.equalizeHist(I)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

I_CLAHE = clahe.apply(I)

cv2.imshow("equalizehist", IGE)
cv2.imshow("CLAHE", I_CLAHE)
cv2.waitKey(0)