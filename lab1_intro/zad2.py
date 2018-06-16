import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

I = plt.imread('mandril.jpg')
plt.figure(1) # stworzenie figury
plt.imshow(I) # dodanie do niej obrazka
plt.title('Mandril') # dodanie tytulu
plt.axis('off') # wylaczenie wyswietlania ukladu wspolrzednych
plt.show() # wyswietlnie calosci

plt.imsave('mandril.png',I)
plt.figure(2)
x = [ 100, 150, 200, 250]
y = [ 50, 100, 150, 200]
plt.plot(x,y,'r.',markersize=10)
plt.show()

fig,ax = plt.subplots(1) # zamiast plt.
rect = plt.Rectangle((50,50),50,100,fill=False, ec='r') # ec - kolor krawedzi
ax.add_patch(rect) # wyswietlenie
plt.axis([40, 100, 40, 100])
plt.show()