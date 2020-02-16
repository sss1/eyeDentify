import cv2
import time

video_dir = '../../data/MOT17_videos/'
def play_experiment_video(video_idx):
  video = cv2.VideoCapture(video_dir + str(video_idx).zfill(2) + '.mp4')

  # Set the inter-frame delay based on the video's natural framerate
  FPS = video.get(cv2.CAP_PROP_FPS) # natural frame rate
  print('Video framerate:' + str(FPS))
  delay = 1.0/FPS

  current_frame = 0
  videoStartTime = time.time()
  nextFrameExists, frame = video.read() # Load first video frame
  
  while nextFrameExists:
    if time.time() > videoStartTime + current_frame * delay:
      cv2.imshow('Video Frame', frame) # Display current frame
      cv2.waitKey(1)
      nextFrameExists, frame = video.read() # Load next video frame
      current_frame += 1
      # TODO: 1) Display object bounding boxes in green
      # 2) Display target object bounding box in red
      # 3) Display gaze point
      # 4) Display estimated object in blue

  video.release()
  cv2.destroyAllWindows()
