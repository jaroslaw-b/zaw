import cv2
import numpy as np

cap = cv2.VideoCapture('test/vid1_IR.avi')
kernel = np.ones((5,5),np.uint8)
iPedestrian = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    G = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, G = cv2.threshold(G, 40, 255, cv2.THRESH_BINARY)
    G = cv2.medianBlur(G, 7)
    G = cv2.morphologyEx(G, cv2.MORPH_CLOSE, kernel)
    output = cv2.connectedComponentsWithStats(G, 4, cv2.CV_32S)
    for o in output[2]:
        if o[cv2.CC_STAT_HEIGHT] > 100 and o[cv2.CC_STAT_AREA] < 10000 :
            cv2.rectangle(frame, (o[cv2.CC_STAT_LEFT], o[cv2.CC_STAT_TOP]), (o[cv2.CC_STAT_LEFT] + o[cv2.CC_STAT_WIDTH], o[cv2.CC_STAT_TOP] + o[cv2.CC_STAT_HEIGHT]), (0, 255, 0), 0)
            roi = frame[o[cv2.CC_STAT_TOP] : o[cv2.CC_STAT_TOP] + o[cv2.CC_STAT_HEIGHT], o[cv2.CC_STAT_LEFT]: o[cv2.CC_STAT_LEFT] + o[cv2.CC_STAT_WIDTH]]
            if iPedestrian%20 == 0:
                cv2.imwrite('wzorce/%06d.png' % int(iPedestrian/20), roi)
            iPedestrian += 1
    cv2.imshow('IR',frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()