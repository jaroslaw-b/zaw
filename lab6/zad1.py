import cv2
import numpy as np
import os
def separate_x_y(c):
    x = c[0][:, 0, 0]
    y = c[0][:, 0, 1]
    M = cv2.moments(c[0])
    x = x - int(M['m10']/M['m00'])
    y = y - int(M['m01']/M['m00'])
    max = -1
    for i in c[0][:,0]:
        for j in c[0][:,0]:
            d = cv2.norm(i, j)
            if d > max:
                max = d
    x = x/max
    y = y/max
    return (x, y, M)

def hausdorf(x1,y1, x2, y2):
    c1 = np.dstack((x1, y1))
    c2 = np.dstack((x2, y2))

    minima1 = list()
    minima2 = list()

    for i in c1[:, 0]:
        for j in c2[:, 0]:
            minima1.append(cv2.norm(i,j))

    dh1 = max(minima1)

    for i in c2[:, 0]:
        for j in c1[:, 0]:
            minima2.append(cv2.norm(i,j))

    dh2 = max(minima2)

    H = max(dh1, dh2)
    return H




img = cv2.imread("plikiHausdorff/imgs/c_astipalea.bmp")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray ^= 255

_, con, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cv2.drawContours(img, con, 0, (0, 255, 0), 1)

[x1, y1, M] = separate_x_y(con)

H = hausdorf(x1,y1,x2,y2)
cv2.imshow("asd", img)
cv2.waitKey(0)


