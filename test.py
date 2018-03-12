import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
from matplotlib.patches import Rectangle
import numpy as np

#openCV
image = cv2.imread('img/mandril.jpg')
image_lena = cv2.imread('img/lena.png')
ig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ihsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lena_gray = cv2.cvtColor(image_lena, cv2.COLOR_BGR2GRAY)
cv2.imshow("Mandril", image)
cv2.imshow("Mandril_grey", ig)
cv2.imshow("Mandril_hsv", ihsv)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('img/result.jpg', image)
print(image.shape)
print(image.size)
print(image.dtype)
i_h = ihsv[:, :, 0]
i_s = ihsv[:, :, 1]
i_v = ihsv[:, :, 2]
cv2.imshow("Mandril_h", i_h)
cv2.imshow("Mandril_s", i_s)
cv2.imshow("Mandril_v", i_v)
cv2.waitKey(0)
cv2.destroyAllWindows()
height,width = image.shape[:2]
scale = 1.75
imagex2 = cv2.resize(image, (int(scale*height), int(scale*width)))
cv2.imshow("image x2", imagex2)
cv2.waitKey(0)
cv2.destroyAllWindows()


add = ig + lena_gray
cv2.imshow("add", add)
cv2.waitKey(0)
cv2.destroyAllWindows()
diff = cv2.absdiff(ig, lena_gray)
cv2.imshow("diff", diff)
cv2.waitKey(0)
cv2.destroyAllWindows()
mult = ig * lena_gray
cv2.imshow("mult", mult)
cv2.waitKey(0)
cv2.destroyAllWindows()

def hist(img):
    h = np.zeros((256,1), np.float32)
    hei, wid = img.shape[:2]
    for y in range(hei):
        for x in range(wid):
            h[img[x, y]] += 1
    return h

h = hist(lena_gray )
plt.plot(h)
plt.show()

_hist = cv2.calcHist([lena_gray], [0], None, [256], [0, 256])
plt.plot(_hist)
plt.show()

ig_e = cv2.equalizeHist(lena_gray)
cv2.imshow("hist_eq", ig_e)
cv2.waitKey(0)
cv2.destroyAllWindows()

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
I_CLAHE = clahe.apply(lena_gray)
cv2.imshow("clahe", I_CLAHE)
cv2.waitKey(0)
cv2.destroyAllWindows()

gauss = cv2.GaussianBlur(lena_gray, (5,5), 3)
cv2.imshow('gauss', gauss)
cv2.waitKey(0)
cv2.destroyAllWindows()

sobel = cv2.Sobel(lena_gray, cv2.CV_64F , 1, 1)
cv2.imshow('sobel', sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()

laplacian = cv2.Laplacian(lena_gray, cv2.CV_64F)
cv2.imshow('laplacian', laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()

median = cv2.medianBlur(lena_gray, 5)
cv2.imshow('median', median)
cv2.waitKey(0)
cv2.destroyAllWindows()
# #Matplotlib
#
# image_1 = plt.imread('img/mandril.jpg')
# fig, ax = plt.subplots(1)
# # plt.figure(1)
# rect = Rectangle((50,50), 50, 100, fill=False, ec='r')
# ax.add_patch(rect)
# plt.imshow(image_1)
# plt.title('Mandril')
# plt.axis('off')
# x = [100, 150, 200, 250]
# y = [50, 100, 150, 200]
# plt.plot(x,y,'r.',markersize=10)
# plt.show()
# plt.imsave('img/mandril.png', image_1)
#
#
# def rgb2gray(I):
#     return 0.299*I[:, :, 0] + 0.587 * I[:, :, 1] + 0.114 * I[:, :, 2]
#
# i_hsv = plt_colors.rgb2hsv(image_1)
# fig, ax = plt.subplots(1)
# plt.gray()
# plt.imshow(i_hsv)
# plt.show()
#
# fig, ax = plt.subplots(1)
# plt.gray()
# image_gray = rgb2gray(image_1)
# plt.imshow(image_gray)
# plt.show()