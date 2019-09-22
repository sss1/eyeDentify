import pickle
import sys, time
from copy import deepcopy
import cv2 # Use OpenCV to display video
import screeninfo
import numpy as np

from centroidtracker import CentroidTracker, calc_centroid

def smooth_objects(all_frames):
  tracker_list = [] 
  # Since we assume that objects cannot change types, we separately run the
  # object tracking algorithm for each object type
  unique_obj_types = set([obj['name'] for frame in all_frames for obj in frame])
  print('Unique object types: ' + str(unique_obj_types))

  # Initialize a centroid tracker for each object type
  trackers = {obj_type : CentroidTracker(maxDisappeared = 15) for obj_type in unique_obj_types}

  for (frame_idx, frame) in enumerate(all_frames):

    new_frame_list = []
    for obj_type in unique_obj_types:
      trackers[obj_type].update([obj['box_points'] for obj in frame if obj['name'] is obj_type]) # Update the centroid tracker

      for (ID, centroid) in trackers[obj_type].objects.items():
        new_ID = obj_type + '_' + str(ID) # Concatenate object type with object ID
        for obj in frame:
          if obj['name'] is obj_type:
          # Since we need to output (ID, bounding box) and Centroid Tracker doesn't record bounding boxes
          # match each ID with its bounding box by centroid; this solution implicitly assumes each object of a
          # specified type within each frame has a distinct centroid
            obj_centroid = calc_centroid(obj['box_points'])
            if (abs(obj_centroid[0] - centroid[0]) < sys.float_info.epsilon) and (abs(obj_centroid[1] - centroid[1]) < sys.float_info.epsilon):
              new_frame_list.append((new_ID, obj['box_points']))

    tracker_list.append(new_frame_list)

    # TODO: Replace numerical obj ID with obj_type + '_' + str(ID) string
  # for (frame_idx, objects) in enumerate(tracker_list):
  #   print('Frame ' + str(frame_idx) + ': ' + str(objects))
  return tracker_list

def compute_durations(tracker_list):
  object_durations = {}
  for (frame_idx, frame_list) in enumerate(tracker_list):
    for obj in frame_list:
      if obj[0] in object_durations:
        object_durations[obj[0]] = (object_durations[obj[0]][0], frame_idx)
      else:
        object_durations[obj[0]] = (frame_idx, frame_idx)
  return object_durations

# This function removes objects from tracker_list if they exist for a shorter time than min_duration frames
def remove_brief_objects(tracker_list, object_durations, min_duration = 15):
  return [[obj for obj in frame_list if
            (object_durations[obj[0]][1] - object_durations[obj[0]][0]) >= min_duration]
            for frame_list in tracker_list]

def sample_targets(tracker_list, object_durations, min_duration = 30, max_duration = 75):
  target_list = []
  next_switch_frame = -1
  for (frame_idx, frame_objects) in enumerate(tracker_list):

    if frame_idx > next_switch_frame:
      # weights = np.array([1 for obj in tracker_list[frame_idx]])
      weights = np.array([object_durations[obj[0]][1] - frame_idx for obj in tracker_list[frame_idx]])
      weights = weights / weights.sum()
      current_target_idx = np.random.choice(range(len(weights)), p = weights)
      current_target = tracker_list[frame_idx][current_target_idx][0]
      print(current_target)
      next_switch_frame = object_durations[current_target][1] # TODO: Potentially shorten this time

    found_target = False
    for obj in tracker_list[frame_idx]:
      print(obj[0], current_target)
      if obj[0] == current_target:
        found_target = True
        target_list.append(obj)
    if not found_target:
      target_list.append(target_list[-1])

  return target_list

def smooth_and_display_objects():

  tracker_list = smooth_objects(all_frames)
  object_durations = compute_durations(tracker_list)
  target_list = sample_targets(tracker_list, object_durations)

  video = cv2.VideoCapture('data/MOT17_videos/13.mp4')

  # Get basic video information
  video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
  video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
  FPS = video.get(cv2.CAP_PROP_FPS) # natural frame rate
  print('Video framerate:' + str(FPS))

  # Set the inter-frame delay based on the video's natural framerate
  delay = 1.0/FPS
  
  # Create video display window and center it on the screen
  screen = screeninfo.get_monitors()[0]
  # (xPos, yPos) is the upper-left corner of the video display
  xPos = int((screen.width - video_width) / 2.0)
  yPos = int((screen.height - video_height) / 2.0)
  if xPos < 0 or yPos < 0:
    print('Warning: video size ' + str((video_width, video_height)) + \
          ' is larger than screen size ' + str((screen.width, screen.height)) + '.')
    print('Some of the video may be cut off.')
  cv2.namedWindow('Video Frame')
  cv2.moveWindow('Video Frame', xPos, yPos)
  
  # Set up to display the first frame
  current_frame = 0
  videoStartTime = time.time()
  videoIsPlaying = True
  nextFrameExists, frame = video.read() # Load first video frame
  
  print("Playing video...")
  while nextFrameExists: # While there are more frames to display, continue displaying video
    if time.time() > videoStartTime + current_frame * delay:

        cv2.imshow('Video Frame', frame) # Display current frame
        cv2.waitKey(1)
        nextFrameExists, frame = video.read() # Load next video frame

        # Draw ellipse around and label target object
        object_ID, b = target_list[current_frame]
        centroid = calc_centroid(b)
        cv2.putText(frame, object_ID, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.ellipse(frame, centroid, ((b[2] - b[0])//2, (b[3] - b[1])//2), 0, 0, 360, color = (0, 255, 0), thickness = 2)

        current_frame += 1

  video.release()
  cv2.destroyAllWindows()

with open('data/detected_objects/13.pickle', 'rb') as in_file:
  all_frames = pickle.load(in_file)

smooth_and_display_objects()
