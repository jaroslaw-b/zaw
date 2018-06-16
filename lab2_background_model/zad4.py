import cv2

I = cv2.imread('sq/office/input/in000001.jpg')
I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
I = cv2.threshold(I, 100, 255, cv2.THRESH_BINARY)
I = I[1]

i_step = 3
i_start = 570
i_stop = 2050
fgbg = cv2.createBackgroundSubtractorMOG2(10, 2, 1)
for i in range(i_start, i_stop, i_step):
    TP_p = 0
    TN_p = 0
    FP_p = 0
    FN_p = 0
    I_clr = cv2.imread('sq/office/input/in00' + str('{0:04}'.format(i)) + '.jpg')
    fgmask = fgbg.apply(I_clr)
    cv2.imshow('frame', fgmask)
    cv2.waitKey(20)