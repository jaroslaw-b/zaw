import cv2

I = cv2.imread('mandril.jpg')
cv2.imshow("Mandril",I) # wyswietlenie
cv2.waitKey(0) # oczekiwanie na klawisz
cv2.destroyAllWindows() # zamkniecie wszystkich okien

cv2.imwrite("m.png",I) # zapis obrazu do pliku

print(I.shape) # rozmiary /wiersze, kolumny, glebia/
print(I.size) # liczba bajtow
print(I.dtype) # typ danych