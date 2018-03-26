import cv2
import numpy as np
import matplotlib.pyplot as plt



W2 = 1
dX = 1
dY = 1
I = cv2.imread('img/I.jpg')
J = cv2.imread('img/J.jpg')

I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
J = cv2.cvtColor(J, cv2.COLOR_RGB2GRAY)

I_diff = cv2.absdiff(I, J)
u = np.zeros(I.shape)
v = np.zeros(I.shape)

for j in range(W2+1, I.shape[0]-2):
    for i in range(W2+1, I.shape[1]-2):
        IO = np.float32(I[j - W2:j + W2 + 1, i - W2:i + W2 + 1])
        dd = np.ones((2*dY+1,2*dX+1),np.float64)*np.inf
        for i2 in range(-dX, dX + 1):
            for j2 in range(-dY, dY + 1):
                JO = np.float32(J[j - W2 + i2:j + W2 + 1 + i2, i - W2 + j2:i + W2 + 1 + j2])
                dd[i2 + dX, j2 + dY] = np.sum(np.sqrt((np.square(JO - IO))))
        ind = np.unravel_index(np.argmin(dd, axis=None), dd.shape)
    # print(ind[0], ind[1])
        u[j, i] = ind[0] -1
        v[j, i] = ind[1] -1

plt.imshow(I)
plt.quiver(u, v, scale = 0.5, scale_units='dots')
plt.show()

# cv2.imshow("I", I)
# cv2.imshow("J", J)
cv2.imshow('I_diff', I_diff)

cv2.waitKey(0)