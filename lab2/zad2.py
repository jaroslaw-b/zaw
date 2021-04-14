import cv2
import numpy as np

I = cv2.imread('sq/office/input/in000001.jpg')
I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
I = cv2.threshold(I, 100, 255, cv2.THRESH_BINARY)
I = I[1]

i_step = 3
i_start = 570
i_stop = 2050

TP = 0
TN = 0
FP = 0
FN = 0
for i in range(i_start, i_stop, i_step):
    TP_p = 0
    TN_p = 0
    FP_p = 0
    FN_p = 0
    I_prev = I
    mask = cv2.imread('sq/office/groundtruth/gt00' + str('{0:04}'.format(i)) + '.png')
    mask = np.uint8(255*(mask==255))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # mask = cv2.threshold(mask[1], 200, 255, cv2.THRESH_BINARY)
    I_clr = cv2.imread('sq/office/input/in00' + str('{0:04}'.format(i)) + '.jpg')
    I = cv2.cvtColor(I_clr, cv2.COLOR_BGR2GRAY)
    I_diff = cv2.absdiff(I, I_prev)
    I_diff = cv2.threshold(I_diff, 35, 255, cv2.THRESH_BINARY)
    I_diff = I_diff[1]
    I_diff = cv2.medianBlur(I_diff, 3)
    I_diff = cv2.dilate(I_diff, cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)), iterations=2)
    I_diff = cv2.erode(I_diff, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) , iterations=2)
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

    # obliczanie parametrów TP TN FP FN
    TP_p = np.logical_and((mask==255),(I_diff==255))
    TN_p = np.logical_and((mask==0),(I_diff==0))
    FP_p = np.logical_and((mask==255),(I_diff==0))
    FN_p = np.logical_and((mask==0),(I_diff==255))
    TP += np.sum(TP_p)
    TN += np.sum(TN_p)
    FP += np.sum(FP_p)
    FN += np.sum(FN_p)

    cv2.imshow("I", I_clr)
    cv2.imshow("mask", mask)
    cv2.waitKey(10)
print(TP, TN, FP, FN)
P = TP/(TP + FP)
R = TP/(TP + FN)
F1 = 2*P*R/(P+R)