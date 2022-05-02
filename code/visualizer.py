import cv2
import math
import numpy as np
import pickle
import time

import hmm
import load_and_preprocess_data
from classes.object_frame import ObjectFrame
import util

SIGMA = 1
TAU = 0.9
VIDEO_DIR = '../data/MOT17_videos/'
DETECTED_OBJECTS_DIR = '../data/detected_objects/'

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1200

# Allow participant 18 frames (300ms) to find new target after switch
GRACE_PERIOD = 18

SAVE_VIDEO_FILENAME = 'output.mp4'

def _rescale_video_to_screen(frame: np.ndarray):
  """Rescale video to fit the screen on which the experiment was displayed."""
  video_height, video_width, _ = frame.shape
  scale = min(SCREEN_HEIGHT/video_height, SCREEN_WIDTH/video_width)
  scaled_width = int(scale * video_width)
  scaled_height = int(scale * video_height)
  frame = cv2.resize(frame, (scaled_width, scaled_height))

  # Pad the remaining screen space with black space
  top_padding = max(0, int((SCREEN_HEIGHT - scaled_height)/2))
  left_padding = max(0, int((SCREEN_WIDTH - scaled_width)/2))
  return cv2.copyMakeBorder(frame, top_padding,
                            SCREEN_HEIGHT - (scaled_height + top_padding),
                            left_padding,
                            SCREEN_WIDTH - (scaled_width + left_padding),
                            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

def _plot_object(frame: np.ndarray, obj: ObjectFrame, color):
  if obj is not None:
    cv2.ellipse(frame, center=obj.centroid, axes=obj.size, angle=0,
                startAngle=0, endAngle=360, color=color, thickness = 2)
  

def play_experiment_video(participant_idx, video_idx, save_video=False):

  video_idx_str = str(video_idx).zfill(2)

  video_fname = VIDEO_DIR + video_idx_str + '.mp4'
  video = cv2.VideoCapture(video_fname)

  detected_objects_fname = DETECTED_OBJECTS_DIR + video_idx_str + '.pickle'
  with open(detected_objects_fname, 'rb') as in_file:
    detected_objects = util.smooth_objects(pickle.load(in_file))
  util.align_objects_to_screen(video_idx, detected_objects)

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

  # Keep track of HMM decoding performance
  hmm_correct = []
  last_switch_frame = 0
  previous_target = experiment_data.frames[0].target

  if save_video:
    out = cv2.VideoWriter(SAVE_VIDEO_FILENAME, -1, FPS/3, (640,480))
  
  while nextFrameExists and current_frame < len(experiment_data.frames):
    if time.time() > videoStartTime + current_frame * delay:
      frame = _rescale_video_to_screen(frame)

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
      hmm_is_correct = (target == hmm_estimate)
      if hmm_is_correct:
        # Plot estimated object in white
        _plot_object(frame, hmm_estimate, color=(255, 255, 255))
      else:
        # Plot target object in red
        _plot_object(frame, target, color=(0, 255, 0))
        # Plot estimated object in green
        _plot_object(frame, hmm_estimate, color=(0, 0, 255))

      if target != previous_target:
        previous_target = target
        last_switch_frame = current_frame
      if current_frame >= last_switch_frame + GRACE_PERIOD:
        # Only count performance on frames outside the grace period
        hmm_correct.append(hmm_is_correct)

      if save_video:
        out.write(frame)
      cv2.imshow('Video Frame', frame) # Display current frame
      cv2.waitKey(1)
      nextFrameExists, frame = video.read() # Load next video frame
      current_frame += 1

  if save_video:
    out.release()
  video.release()
  cv2.destroyAllWindows()

  print('HMM accuracy: {}'.format(np.mean(hmm_correct)))


if __name__ == '__main__':
  play_experiment_video(participant_idx=16, video_idx=1, save_video=True)
