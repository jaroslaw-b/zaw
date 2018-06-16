import cv2


I = cv2.imread('mandril.jpg')

height, width =I.shape[:2] # pobranie elementow 1 i 2 tj. odpowienio wysokosci i szerokosci
scale = 1.75 # wspolczynnik skalujacy
Ix2 = cv2.resize(I,(int(scale*height),int(scale*width)))
cv2.imshow("Big Mandril",Ix2)
cv2.waitKey(0)