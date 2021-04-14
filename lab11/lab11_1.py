import cv2

import math                                     # do PI
from mpl_toolkits.mplot3d import Axes3D         # do 3D
from matplotlib import pyplot as plt
import numpy as np



# Generowanie Gaussa
kernel_size = 75                            # rozmiar rozkladu
sigma = 10                           # odchylenie std
x = np.arange(0, kernel_size, 1, float)     # wektor poziomy
y = x[:,np.newaxis]                  # wektor pionowy
x0 = y0 = kernel_size // 2                  # wsp. srodka
G = 1/(2*math.pi*sigma**2)*np.exp(-0.5*((x-x0)**2 + (y-y0)**2) / sigma**2)
# Rysowanie
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, G, color='b')
# plt.show()


G_y = np.diff(G,1,0)
G_y = np.append(G_y,np.zeros((1,kernel_size), float), 0)    # dodaniedodatkowego wiersza
G_y = -G_y
G_x = np.diff(G,1,1)
G_x = np.append(G_x,np.zeros((kernel_size,1), float), 1)    # dodaniedodatkowej kolumny
G_x = -G_x

kernel_size = 45                            # rozmiar rozkladu
def track_init(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.rectangle(I, (x-kernel_size//2, y- kernel_size//2), (x + kernel_size, y + kernel_size), (0, 255, 0), 2)
    mouseX,mouseY = x,y
    # Wczytanie pierwszego obrazka

id = 100
I = cv2.imread('track_seq/track00'+str(id)+'.png')
cv2.namedWindow('Tracking')
cv2.setMouseCallback('Tracking',track_init)
hist_q = np.zeros((256, 1), float)
hist_p = np.zeros((256, 1), float)
I_HSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
I_H = I_HSV[:, :, 0]
# Pobranie klawisza
cv2.imshow('Tracking',I )
k = cv2.waitKey(0) & 0xFF
xS = mouseX-kernel_size//2
yS = mouseY-kernel_size//2
xC = xS
yC = yS
for jj in range(0, kernel_size):
    for ii in range(0, kernel_size):
        pixel_H = I_H[yS + jj, xS + ii]
        hist_p[pixel_H] = hist_p[pixel_H] + pixel_H * G[jj, ii]

while 1:
    cv2.rectangle(I, (xC, yC ), (xC + kernel_size, yC + kernel_size), (0, 255, 0), 2)
    cv2.imshow('Tracking',I )
    k = cv2.waitKey(20) & 0xFF
    if k == 27:   # ESC
        break

    xS = xC
    yS = yC
    for jj in range(0, kernel_size):
        for ii in range(0, kernel_size):
            pixel_H = I_H[yC + jj, xC + ii]
            hist_q[pixel_H] = hist_q[pixel_H] + pixel_H*G[jj, ii]


    dx_l = 0
    dx_m = 0
    dy_l = 0
    dy_m = 0

    for k in range(20):
        rho = np.sqrt(hist_q*hist_p)
        for jj in range(0,kernel_size):
            for ii in range(0,kernel_size):
                dx_l = dx_l + ii*rho[I_H[yC+jj,xC+ii]]*G_x[jj,ii]
                dx_m = dx_m + ii*G_x[jj,ii]
                dy_l = dy_l + jj*rho[I_H[yC+jj,xC+ii]]*G_y[jj,ii]
                dy_m = dy_m + jj*G_y[jj,ii]
        dx = dx_l/dx_m
        dy = dy_l/dy_m
        # Obliczanie nowych wspolrzednych
        xC = np.int(np.floor(xC + dx))
        yC = np.int(np.floor(yC + dy))
    id += 1
    I = cv2.imread('track_seq/track00' + str(id) + '.png')
    I_HSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    I_H = I_HSV[:, :, 0]