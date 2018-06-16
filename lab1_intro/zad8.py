import cv2

I = cv2.imread('lena.png', 0)

I_gauss = cv2.GaussianBlur(I, (3, 3), 1)
I_sobel = cv2.Sobel(I,cv2.CV_64F,1,0,ksize=5)
I_laplacian = cv2.Laplacian(I, cv2.CV_64F)
I_median = cv2.medianBlur(I, 5)

cv2.imshow("Gauss", I_gauss)
cv2.imshow("Sobel", I_sobel)
cv2.imshow("Laplacian", I_laplacian)
cv2.imshow("Median", I_median)
cv2.waitKey(0)