import pickle, sys, time
from copy import deepcopy
import cv2 # Use OpenCV to display video
import numpy as np

from centroidtracker import CentroidTracker, calc_centroid

horz_scale = 0.8
vert_scale = 0.8

def _smooth_objects(all_frames):
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

  return tracker_list

def _compute_durations(tracker_list):
  object_durations = {}
  for (frame_idx, frame_list) in enumerate(tracker_list):
    for obj in frame_list:
      if obj[0] in object_durations:
        object_durations[obj[0]] = (object_durations[obj[0]][0], frame_idx)
      else:
        object_durations[obj[0]] = (frame_idx, frame_idx)
  return object_durations

# # This function removes objects from tracker_list if they will exist for fewer than min_duration frames past current_frame
# def remove_brief_objects(tracker_list, object_durations, current_frame, min_duration = 15):
#   return [[obj for obj in frame_list if
#             (object_durations[obj[0]][1] - object_durations[obj[0]][0]) >= min_duration]
#             for frame_list in tracker_list]

def _interpolate_missing_frames(target_list):
  prev_good_frame = 0
  for frame_idx in range(len(target_list)):
    if prev_good_frame < frame_idx - 1: # if previous frames were missing
      if target_list[frame_idx] is not None: # if current frame is non-missing, we need to interpolate till here

        # Objects should only change on non-missing frames
        target_name = target_list[prev_good_frame][0]
        if target_name == target_list[frame_idx][0]:

          # Get first and last non-missing boxes to interpolate between
          prev_good_box = target_list[prev_good_frame][1]
          current_box = target_list[frame_idx][1]

          num_parts = frame_idx - prev_good_frame
          for frame_to_interpolate in range(prev_good_frame + 1, frame_idx):
            interpolation_idx = frame_to_interpolate - prev_good_frame
            interpolate = lambda x, y : int(x + interpolation_idx/num_parts * (y - x))
            interpolated_box = (interpolate(prev_good_box[0], current_box[0]),
                                interpolate(prev_good_box[1], current_box[1]),
                                interpolate(prev_good_box[2], current_box[2]),
                                interpolate(prev_good_box[3], current_box[3]))
            target_list[frame_to_interpolate] = (target_name, interpolated_box)

    if target_list[frame_idx] is not None:
      # All frames before (and including) frame_idx have been filled in
      prev_good_frame = frame_idx

  return target_list

def _sample_targets(tracker_list, object_durations, min_duration = 30, mean_duration = 45):
  target_list = []
  next_switch_frame = -1
  for (frame_idx, frame_objects) in enumerate(tracker_list):

    if frame_idx > next_switch_frame:
      # Weight each object by its remaining duration to prefer longer-lasting objects
      weights = np.array([object_durations[obj[0]][1] - frame_idx for obj in tracker_list[frame_idx]])
      if np.sum(weights) < sys.float_info.epsilon: # No objects were found in tracker_list
        target_list.append(None)
        continue
      weights = weights / weights.sum()
      current_target_idx = np.random.choice(range(len(weights)), p = weights)
      current_target = tracker_list[frame_idx][current_target_idx][0]
      next_switch_frame = object_durations[current_target][1]
      next_switch_frame = min(next_switch_frame, frame_idx + min_duration + int(np.random.exponential(mean_duration)))
      attempt = 0
      while current_target not in [obj[0] for obj in tracker_list[next_switch_frame]]:
        attempt += 1
        if attempt % 100 == 0:
          print(attempt)
        if attempt > 1000:
          # For some objects/frames, we may not be able to find a valid next_swith_frame even after many attempts.
          # In this case, simply switch on the very next frame
          next_switch_frame = frame_idx
          break
        next_switch_frame = min(next_switch_frame, frame_idx + min_duration + int(np.random.exponential(mean_duration)))
      # However, we need to check that we only switch on non-missing frames

    found_target = False
    for obj in tracker_list[frame_idx]:
      if obj[0] == current_target:
        found_target = True
        target_list.append(obj)
    if not found_target:
      target_list.append(None)

  return _interpolate_missing_frames(target_list)

def smooth_and_display_objects(video_idx):

  # Load object data, smooth over time, and select a sequence of targets
  with open('data/detected_objects/' + str(video_idx).zfill(2) + '.pickle', 'rb') as in_file:
    all_frames = pickle.load(in_file)
  tracker_list = _smooth_objects(all_frames)
  object_durations = _compute_durations(tracker_list)
  target_list = _sample_targets(tracker_list, object_durations)

  video = cv2.VideoCapture('data/MOT17_videos/' + str(video_idx).zfill(2) + '.mp4')

  # Get basic video information
  FPS = video.get(cv2.CAP_PROP_FPS) # natural frame rate
  print('Video framerate:' + str(FPS))

  # Set the inter-frame delay based on the video's natural framerate
  delay = 1.0/FPS
  
  # Create video display window and center it on the screen
  cv2.namedWindow('Video Frame')
  
  # Set up to display the first frame
  current_frame = 0
  videoStartTime = time.time()
  videoIsPlaying = True
  nextFrameExists, frame = video.read() # Load first video frame
  height, width, _ =  frame.shape
  frame = cv2.resize(frame, (int(horz_scale * width), int(vert_scale * height))) # rescale by horz_scale * vert_scale
  
  print("Playing video...")
  while nextFrameExists: # While there are more frames to display, continue displaying video
    if time.time() > videoStartTime + current_frame * delay:

        cv2.imshow('Video Frame', frame) # Display current frame
        cv2.waitKey(1)
        nextFrameExists, frame = video.read() # Load next video frame
        if nextFrameExists:
          frame = cv2.resize(frame, (int(horz_scale * width), int(vert_scale * height))) # rescale by horz_scale * vert_scale

          # Draw ellipse around and label target object
          if target_list[current_frame] is not None: # This happens as long as there is at least one detected object on screen
            object_ID, b = target_list[current_frame]
            centroid = calc_centroid(b)
            centroid = (int(horz_scale * centroid[0]), int(vert_scale * centroid[1]))
            cv2.putText(frame, object_ID, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.ellipse(frame, centroid, (int(horz_scale * (b[2] - b[0])/2), int(vert_scale * (b[3] - b[1])/2)), 0, 0, 360, color = (0, 255, 0), thickness = 2)

        current_frame += 1

  video.release()
  cv2.destroyAllWindows()

smooth_and_display_objects(video_idx = 9)
