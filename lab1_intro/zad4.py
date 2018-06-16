import matplotlib.pyplot as plt

def rgb2gray(I):
    return 0.299*I[:,:,0] + 0.587*I[:,:,1] + 0.114*I[:,:,2]

I = plt.imread('mandril.jpg')

IG = rgb2gray(I)

plt.imshow(IG)
plt.gray()
plt.show()