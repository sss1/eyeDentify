import cv2
import math
import pickle
import time

import load_and_preprocess_data
import util

VIDEO_DIR = '../../data/MOT17_videos/'
DETECTED_OBJECTS_DIR = '../../data/detected_objects/'

def play_experiment_video(participant_idx, video_idx):

  video_idx_str = str(video_idx).zfill(2)

  video_fname = VIDEO_DIR + video_idx_str + '.mp4'
  video = cv2.VideoCapture(video_fname)

  detected_objects_fname = DETECTED_OBJECTS_DIR + video_idx_str + '.pickle'
  with open(detected_objects_fname, 'rb') as in_file:
    detected_objects = util.smooth_objects(pickle.load(in_file))

  experiment_data = (load_and_preprocess_data
                     .load_participant(participant_idx)
                     .videos[video_idx-1])

  # Set the inter-frame delay based on the video's natural framerate
  FPS = video.get(cv2.CAP_PROP_FPS) # natural frame rate
  print('Video framerate:' + str(FPS))
  delay = 1.0/FPS

  current_frame = 0
  videoStartTime = time.time()
  nextFrameExists, frame = video.read() # Load first video frame
  
  while nextFrameExists and current_frame < len(experiment_data.frames):
    if time.time() > videoStartTime + current_frame * delay:

      experiment_frame = experiment_data.frames[current_frame] 
      # Plot gaze
      try:
        gaze = tuple(int(x) for x in experiment_frame.gaze)
        image = cv2.circle(frame, center=gaze, radius=10, color=(255, 255, 255),
                           thickness = 3)
      except ValueError: # Skip eye-tracking when data is missing
        pass

      # Plot target object
      target_centroid = experiment_frame.target.centroid
      target_size = experiment_frame.target.size
      cv2.ellipse(frame, center=target_centroid,
                  axes=(target_size[0], target_size[1]), angle=0, startAngle=0,
                  endAngle=360, color = (0, 255, 0), thickness = 2)
      cv2.imshow('Video Frame', frame) # Display current frame

      cv2.waitKey(1)
      nextFrameExists, frame = video.read() # Load next video frame
      current_frame += 1
      # TODO: 1) Display object bounding boxes in green
      # 2) Display estimated object in blue

  video.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  play_experiment_video(0, 1)
