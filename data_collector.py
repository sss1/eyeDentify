import numpy as np
import cv2
import time

video = cv2.VideoCapture('traffic-mini.mp4')

current_frame = 0
video_start_time = time.time()

FPS = video.get(cv2.CAP_PROP_FPS)
delay = 1.0/FPS

while(video.isOpened()):
    ret, frame = video.read() # Load next frame
    if not ret: # If done playing video
      break

    cv2.imshow('frame',frame) # Display frame

    current_frame += 1
    while time.time() < video_start_time + current_frame * delay:
      cv2.waitKey(1)

print('Runtime: ' + str(time.time() - video_start_time))

video.release()
cv2.destroyAllWindows()
