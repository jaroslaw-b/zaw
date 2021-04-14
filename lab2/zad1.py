import cv2
import numpy as np

I = cv2.imread('input/in000301.jpg')
I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
I = cv2.threshold(I, 100, 255, cv2.THRESH_BINARY)
I = I[1]



for i in range(1, 150):
    I_prev = I
    I_clr = cv2.imread('input/in000' + str(300 + i) + '.jpg')
    I = cv2.cvtColor(I_clr, cv2.COLOR_BGR2GRAY)
    I_diff = cv2.absdiff(I, I_prev)
    I_diff = cv2.threshold(I_diff, 40, 255, cv2.THRESH_BINARY)
    I_diff = I_diff[1]
    I_diff = cv2.medianBlur(I_diff, 3)
    I_diff = cv2.dilate(I_diff, cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)), iterations=3)
    I_diff = cv2.erode(I_diff, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(I_diff)
    # cv2.imshow("Labels", np.uint8(labels / stats.shape[0]*255))
    if (stats.shape[0] > 1):  # czy sa jakies obiekty
        pi, p = max(enumerate(stats[1:, 4]), key=(lambda x: x[1]))
        pi = pi + 1
        # wyrysownie bbox
        cv2.rectangle(I_clr, (stats[pi, 0], stats[pi, 1]), (stats[pi, 0] + stats[pi , 2], stats[pi, 1] + stats[pi, 3]), (255, 0, 0), 2)
        # wypisanie informacji
        cv2.putText(I_clr, "%f" % stats[pi, 4], (stats[pi, 0], stats[pi, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        cv2.putText(I_clr, "%d" % pi, (np.int(centroids[pi, 0]), np.int(centroids[pi, 1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))



    cv2.imshow("I", I_clr)
    cv2.waitKey(10)
