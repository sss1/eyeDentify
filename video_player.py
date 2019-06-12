import numpy as np
import cv2
import time

cap = cv2.VideoCapture('traffic-mini.mp4')

start = time.time()

while(cap.isOpened()):
    ret, frame = cap.read() # Load next frame

    if not ret: # Done playing video
      break

    cv2.imshow('frame',frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

print('Runtime: ' + str(time.time() - start))

cap.release()
cv2.destroyAllWindows()
