import cv2
import numpy as np
img = cv2.imread('img/trybik.jpg')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, img_bin = cv2.threshold(img_gray, 235, 255, cv2.THRESH_BINARY_INV)

sobelx = cv2.Sobel(img_bin,cv2.CV_64F,1,0,ksize=5)

sobely = cv2.Sobel(img_bin,cv2.CV_64F,0,1,ksize=5)

sobel_xy = (sobelx**2 + sobely**2)**0.5
sobel_max = np.max(sobel_xy)
sobel_xy = sobel_xy/sobel_max
sobel_orient = np.arctan2(sobely, sobelx)

M = cv2.moments(img_bin)
m_x = int(M['m10'] / M['m00'])
m_y = int(M['m01'] / M['m00'])

_, con, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

r_table =  [[] for i in range(360)]

for c in con[0]:
    v = np.array([c[0][0] - m_x, c[0][1] - m_y])
    angle = np.arctan2(v[1], v[0])
    r_table_address = int(sobel_orient[tuple(c[0])] * 360 /(2*np.pi) + 180)
    if r_table_address == 360:
        r_table_address = 0
    object2append = [cv2.norm(v), angle]
    r_table[r_table_address].append(object2append)


img2 = cv2.imread('img/trybiki2.jpg')

img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

_, img_bin2 = cv2.threshold(img_gray2, 200, 255, cv2.THRESH_BINARY_INV)

sobelx2 = cv2.Sobel(img_bin2,cv2.CV_64F,1,0,ksize=5)

sobely2 = cv2.Sobel(img_bin2,cv2.CV_64F,0,1,ksize=5)

sobel_xy2 = (sobelx2**2 + sobely2**2)**0.5
sobel_max2 = np.max(sobel_xy2)
sobel_xy2 = sobel_xy2/sobel_max2
sobel_orient2 = np.arctan2(sobely2, sobelx2)

h = img2.shape[0]
w = img2.shape[1]
hough_space = np.zeros((2*h, 2*w))

for i in range(h):
    for j in range(w):
        if sobel_xy2[i][j] > 0.5:
            for el in r_table[int(sobel_orient2[i][j])]:
                r = el[0]
                fi = el[1]
                x1 = r * np.cos(fi) + i
                y1 = r * np.sin(fi) + j
                hough_space[int(round(y1)), int(round(x1))] += 1


for iter in range(15):
    max = np.unravel_index(np.argmax(hough_space), hough_space.shape)
    cv2.circle(img2, max, 2, (0, 0, 255), 4)
    hough_space[max] = 0


cv2.drawContours(img, con, 0, (0, 255, 0), 1)


cv2.imshow("Binary", img2)
cv2.waitKey(0)