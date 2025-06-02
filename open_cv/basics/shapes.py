import cv2 as cv
import numpy as np


blank = np.ones((500, 500, 3))
cv.rectangle(blank, (0,0), (250, 250), (0, 255, 0), thickness=1)
cv.imshow("blank", blank)

cv.circle(blank, (blank.shape[0]//2, blank.shape[1]//2), 40, (255, 255, 0), thickness=cv.FILLED)
cv.imshow("blank", blank)

cv.putText(blank, "Hello Churchil", (255, 255), cv.FONT_HERSHEY_TRIPLEX,
           1.0, (0, 0,0), 2)

cv.imshow("blank", blank)
cv.waitKey(0)

