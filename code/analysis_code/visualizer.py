import cv2
import math
import numpy as np
import pickle
import time

import hmm
import load_and_preprocess_data
import object_frame
import util

SIGMA = 1
TAU = 0.9
VIDEO_DIR = '../../data/MOT17_videos/'
DETECTED_OBJECTS_DIR = '../../data/detected_objects/'

def _plot_object(frame: np.ndarray, obj: object_frame.ObjectFrame, color):
  if obj is not None:
    cv2.ellipse(frame, center=obj.centroid, axes=obj.size, angle=0,
                startAngle=0, endAngle=360, color=color, thickness = 2)
  

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

  hmm_mle = hmm.forwards_backwards(SIGMA, TAU, experiment_data,
                                   detected_objects)

  # Set the inter-frame delay based on the video's natural framerate
  FPS = video.get(cv2.CAP_PROP_FPS) # natural frame rate
  print('Video framerate:' + str(FPS))
  delay = 1.0/FPS

  current_frame = 0
  videoStartTime = time.time()
  nextFrameExists, frame = video.read() # Load first video frame

  hmm_correct = np.zeros((len(hmm_mle),), dtype=bool)
  
  while nextFrameExists and current_frame < len(experiment_data.frames):
    if time.time() > videoStartTime + current_frame * delay:

      # Plot gaze
      try:
        gaze = tuple(int(x) for x in experiment_data.frames[current_frame].gaze)
        cv2.circle(frame, center=gaze, radius=10, color=(255, 255, 255),
                   thickness = 3)
      except ValueError:
        # When eye-tracking is missing, plot a red square in the top-left corner
        cv2.rectangle(frame, (0, 0), (20, 20), (0, 0, 255), 20)
        pass

      # Plot all detected objects
      objects_in_frame = detected_objects[current_frame]
      for obj in objects_in_frame:
        _plot_object(frame, obj, color=(255, 0, 0))

      # Plot true target and HMM estimate
      target = experiment_data.frames[current_frame].target
      hmm_estimate = hmm_mle[current_frame]
      if target != hmm_estimate:
        # Plot target object in red
        _plot_object(frame, target, color=(0, 0, 255))
        # Plot estimated object in green
        _plot_object(frame, hmm_estimate, color=(0, 255, 0))
      else:
        # Plot estimated object in white
        _plot_object(frame, hmm_estimate, color=(255, 255, 255))
        hmm_correct[current_frame] = True

      cv2.imshow('Video Frame', frame) # Display current frame
      cv2.waitKey(1)
      nextFrameExists, frame = video.read() # Load next video frame
      current_frame += 1

  video.release()
  cv2.destroyAllWindows()

  print('HMM accuracy: {}'.format(hmm_correct.mean()))


if __name__ == '__main__':
  play_experiment_video(0, 1)
