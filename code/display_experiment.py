import csv
import cv2 # Use OpenCV to display video
from datetime import datetime
import numpy as np
import object_smoothing as object_smoothing
import pickle
import random
import time
import util

# for simplicity, in the actual experiment, we hard-coded these values
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1200
_TARGET_COLOR = (0, 255, 0) # bright green
DETECTED_OBJECTS_DIR = '../data/detected_objects/multifidelity/'
VIDEO_DIR = '../data/MOT17_videos/'

def smooth_and_display_objects(video_idx, confidence_threshold):

  detected_objects_fname = 'video{}_threshold{}.pickle'.format(
          str(video_idx).zfill(2), confidence_threshold)
  video_fname = str(video_idx).zfill(2) + '.mp4'
  print(VIDEO_DIR + video_fname)

  print('Playing video {} at confidence {}'
        .format(video_idx, confidence_threshold))
  
  # Load object data, smooth over time, and select a sequence of targets
  with open(DETECTED_OBJECTS_DIR + detected_objects_fname, 'rb') as in_file:
    all_frames = pickle.load(in_file)
  target_list = object_smoothing.generate_target_list(all_frames)

  video = cv2.VideoCapture(VIDEO_DIR + video_fname)

  # Get basic video information
  FPS = video.get(cv2.CAP_PROP_FPS) # natural frame rate
  print('Video framerate:' + str(FPS))

  # Set the inter-frame delay based on the video's natural framerate
  delay = 1.0/FPS
  
  # Set up to display the first frame
  current_frame = 0
  videoStartTime = time.time()
  videoIsPlaying = True
  nextFrameExists, frame = video.read() # Load first video frame
  video_height, video_width, _ = frame.shape

  # Recale video to be as large as possible while fitting on screen without
  # changing aspect ratio
  scale = min(SCREEN_HEIGHT/video_height, SCREEN_WIDTH/video_width)
  scaled_width = int(scale * video_width)
  scaled_height = int(scale * video_height)
  frame = cv2.resize(frame, (scaled_width, scaled_height)) # rescale video to fit screen

  # Pad the remaining screen space with black space
  top_padding = max(0, int((SCREEN_HEIGHT - scaled_height)/2))
  left_padding = max(0, int((SCREEN_WIDTH - scaled_width)/2))
  border_color = (0, 0, 0) # black border
  frame = cv2.copyMakeBorder(frame,
                             top_padding,
                             SCREEN_HEIGHT - (scaled_height + top_padding),
                             left_padding,
                             SCREEN_WIDTH - (scaled_width + left_padding),
                             borderType=cv2.BORDER_CONSTANT,
                             value=border_color)
                     
  timestamped_target_list = [] # List of timestamped targets to output
  current_time = centroid = object_ID = horz_rad = \
    vert_rad = target_conf = None # all frame-specific variables to output
   
  print("Playing video...")

  # Used to that eye-tracker and video are spatially calibrated
  fixation_point = None
  print('Fixation Coordinates:', fixation_point)

  # While there are more frames to display, continue displaying video
  while nextFrameExists:
    current_time = time.time()
    if current_time > videoStartTime + current_frame * delay:
        cv2.imshow('Video Frame', frame) # Display current frame
        
        # Record target and timestamp if target exists
        if centroid is not None:
          timestamped_target_list.append([current_time,
                                          video_idx,
                                          confidence_threshold,
                                          object_ID,
                                          target_conf,
                                          centroid[0],
                                          centroid[1],
                                          horz_rad,
                                          vert_rad])
          
        nextFrameExists, frame = video.read() # Load next video frame
        cv2.waitKey(1)
        
        if nextFrameExists:
          frame = cv2.resize(frame, (scaled_width, scaled_height)) # rescale video to fit screen
          frame = cv2.copyMakeBorder(frame, top_padding, top_padding,
                                     left_padding, left_padding,
                                     borderType=cv2.BORDER_CONSTANT, value=0)

          # Draw ellipse around and label target object
          if target_list[current_frame] is not None: # This happens as long as there is at least one detected object on screen
            object_ID, b, target_conf = target_list[current_frame]
            centroid = util.calc_centroid(b)
            centroid = (int(scale * centroid[0]) + left_padding, int(scale * centroid[1]) + top_padding)
            horz_rad = int(scale * (b[2] - b[0])/2)
            vert_rad = int(scale * (b[3] - b[1])/2)
            cv2.ellipse(frame, centroid, (horz_rad, vert_rad), 0, 0, 360, color = _TARGET_COLOR, thickness = 2)
          if fixation_point is not None:
            cv2.ellipse(frame, fixation_point, (10, 10), 0, 0, 360, color = (0, 255, 0), thickness = 3)

        current_frame += 1

  video.release()
  return timestamped_target_list
# Display the 14 videos in random order
print('\nThis is the stimulus display script.\n\n')
print('Start this 2nd!\n\n')
participant_id = input('Enter participant ID: ')

# Construct output file path
today = '{}-{}-{}'.format(datetime.now().month, datetime.now().day,
                          datetime.now().year)
time_now = '{}_{}'.format(datetime.now().hour, datetime.now().minute)
file_name = '{}_stimulus_{}_{}.csv'.format(participant_id, today, time_now)
print('Outputting stimulus data to ' + file_name)
with open(file_name, 'w') as outfile:

  writer = csv.writer(outfile, delimiter = ',')
  title = ["Participant ID: " + participant_id, "Date: " + today,
           "Time: " + time_now]
  heading = ["ComputerClock_Timestamp", "Video_Index",
             "Object_Detection_Threshold", "Target_Name", "Target_Confidence",
             "TargetX", "TargetY", "TargetXRadius", "TargetYRadius"]
  writer.writerow(title)
  writer.writerow(heading)

  # Create fullscreen video display window
  black_screen = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT, 3))
  cv2.line(black_screen,
           (int(SCREEN_HEIGHT/2)-10, int(SCREEN_WIDTH/2)),
           (int(SCREEN_HEIGHT/2)+10, int(SCREEN_WIDTH/2)),
           (255, 255, 255), 3, cv2.LINE_AA)
  cv2.line(black_screen,
           (int(SCREEN_HEIGHT/2), int(SCREEN_WIDTH/2)-20),
           (int(SCREEN_HEIGHT/2), int(SCREEN_WIDTH/2)+20),
           (255, 255, 255), 2, cv2.LINE_AA)
  cv2.namedWindow('Video Frame', cv2.WND_PROP_FULLSCREEN)
  cv2.setWindowProperty('Video Frame', cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN)
  cv2.imshow('Video Frame', black_screen)
  cv2.waitKey(5000)

  # Display the stimulus videos in a random order, with 5 second breaks
  video_sequence = ([(video_idx, confidence_threshold)
                     for video_idx in range(1, 15)
                     for confidence_threshold in [40,60,80]])
  random.shuffle(video_sequence)
  
  for (video_idx, confidence_threshold) in video_sequence:

    # Skip video 1 at conf 80 because no objects were detected
    if video_idx == 1 and confidence_threshold == 80:
      continue
    
    # Display next video and record target data
    output = smooth_and_display_objects(
        video_idx=video_idx, confidence_threshold=confidence_threshold)

    # Display black screen for 5 seconds between each video
    cv2.imshow('Video Frame', black_screen)
    cv2.waitKey(5000)

    # Write trial output
    for row in output:
      writer.writerow(row)

  cv2.destroyAllWindows()
