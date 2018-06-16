import cv2

I = cv2.imread('mandril.jpg')

IG = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
IHSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)

IH = IHSV[:,:,0]
IS = IHSV[:,:,1]
IV = IHSV[:,:,2]

cv2.imshow("H", IH)
cv2.imshow("S", IS)
cv2.imshow("V", IV)
cv2.imshow("IG", IG)
cv2.imshow("HSV", IHSV)

cv2.waitKey(0)