import pickle, sys, time, csv, random
from datetime import datetime
from copy import deepcopy
import cv2 # Use OpenCV to display video
import numpy as np

from centroidtracker import CentroidTracker, calc_centroid

# For simplicity, we hard-coded these values
screen_width = 1920
screen_height = 1200

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
      # Update the centroid tracker
      trackers[obj_type].update([obj['box_points'] for obj in frame if obj['name'] is obj_type])

      for (ID, centroid) in trackers[obj_type].objects.items():
        new_ID = obj_type + '_' + str(ID) # Concatenate object type with object ID
        for obj in frame:
          if obj['name'] is obj_type:
          # Since we need to output (ID, bounding box) and Centroid Tracker doesn't record bounding boxes
          # match each ID with its bounding box by centroid; this solution implicitly assumes each object of a
          # specified type within each frame has a distinct centroid
            obj_centroid = calc_centroid(obj['box_points'])
            if (abs(obj_centroid[0] - centroid[0]) < sys.float_info.epsilon) and \
                 (abs(obj_centroid[1] - centroid[1]) < sys.float_info.epsilon):
              new_frame_list.append((new_ID, obj['box_points'], obj['percentage_probability']))

    tracker_list.append(new_frame_list)

  return tracker_list

def _compute_durations(tracker_list):
  """Computes object_durations, a dict mapping from object ID strings to
     (first_frame, last_frame) pairs indicating the the object first and last
     appears
  """
  object_durations = {}
  for (frame_idx, frame_list) in enumerate(tracker_list):
    for obj in frame_list:
      if obj[0] in object_durations:
        object_durations[obj[0]] = (object_durations[obj[0]][0], frame_idx)
      else:
        object_durations[obj[0]] = (frame_idx, frame_idx)
  return object_durations

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
            target_list[frame_to_interpolate] = (target_name, interpolated_box, float('nan'))

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
      next_switch_frame = min(next_switch_frame,
                              frame_idx + min_duration + int(np.random.exponential(mean_duration)))
      attempt = 0
      while current_target not in [obj[0] for obj in tracker_list[next_switch_frame]]:
        attempt += 1
        # if attempt % 100 == 0:
        #   print(attempt)
        if attempt > 1000:
          # For some objects/frames, we may not be able to find a valid next_swith_frame even after many attempts.
          # In this case, simply switch on the very next frame
          next_switch_frame = frame_idx
          break
        next_switch_frame = min(next_switch_frame,
                                frame_idx + min_duration + int(np.random.exponential(mean_duration)))

    # However, we need to check that we only switch on non-missing frames
    found_target = False
    for obj in tracker_list[frame_idx]:
      if obj[0] == current_target:
        found_target = True
        target_list.append(obj)
    if not found_target:
      target_list.append(None)

  return _interpolate_missing_frames(target_list)

data_dir = 'C:/Users/infant lab user/Desktop/eyeDentify/data/'

def smooth_and_display_objects(video_idx, confidence_threshold):

  detected_objects_fname = \
    'video{idx}_threshold{conf}.pickle'.format(idx=str(video_idx).zfill(2),
                                               conf=confidence_threshold)
  video_fname = str(video_idx).zfill(2) + '.mp4'

  print('Playing video {idx} at confidence {conf}'.format(idx=video_idx,
                                                          conf=confidence_threshold))
  
  # Load object data, smooth over time, and select a sequence of targets
  with open(data_dir + 'detected_objects/' + detected_objects_fname, 'rb') as in_file:
    all_frames = pickle.load(in_file)
  tracker_list = _smooth_objects(all_frames)
  object_durations = _compute_durations(tracker_list)
  target_list = _sample_targets(tracker_list, object_durations)

  video = cv2.VideoCapture(data_dir + 'MOT17_videos/' + video_fname)

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
  video_height, video_width, _ =  frame.shape

  # Recale video to be as large as possible while fitting on screen without changing aspect ratio
  scale = min(screen_height/video_height, screen_width/video_width)
  scaled_width = int(scale * video_width)
  scaled_height = int(scale * video_height)
  frame = cv2.resize(frame, (scaled_width, scaled_height)) # rescale video to fit screen

  # Pad the remaining screen space with black space
  top_padding = max(0, int((screen_height - scaled_height)/2))
  left_padding = max(0, int((screen_width - scaled_width)/2))
  border_color = (0, 0, 0) # black border
  frame = cv2.copyMakeBorder(frame, top_padding, screen_height - (scaled_height + top_padding),
                             left_padding, screen_width - (scaled_width + left_padding),
                             borderType=cv2.BORDER_CONSTANT, value=border_color)
                     
  timestamped_target_list = [] # List of timestamped targets to output
  current_time = centroid = object_ID = horz_rad = \
    vert_rad = target_conf = None # all the frame-specific variables we'll be outputting
   
  print("Playing video...")

  # Used to that eye-tracker and video are spatially calibrated
  fixation_point = None #(int(3*screen_width/4), int(3*screen_height/4))
  print('Fixation Coordinates:', fixation_point)

  while nextFrameExists: # While there are more frames to display, continue displaying video
    current_time = time.time()
    if current_time > videoStartTime + current_frame * delay:
        cv2.imshow('Video Frame', frame) # Display current frame
        
        # Record target and timestamp if target exists
        if centroid is not None:
          timestamped_target_list.append([current_time,
                                          video_idx,
                                          confidence_threshold, # Threshold for this stimulus
                                          object_ID,
                                          target_conf, # Confidence of this specific target
                                          centroid[0],
                                          centroid[1],
                                          horz_rad,
                                          vert_rad])
          
        nextFrameExists, frame = video.read() # Load next video frame
        cv2.waitKey(1)
        
        if nextFrameExists:
          frame = cv2.resize(frame, (scaled_width, scaled_height)) # rescale video to fit screen
          frame = cv2.copyMakeBorder(frame, top_padding, top_padding, left_padding, left_padding,
                                     borderType=cv2.BORDER_CONSTANT, value=0)

          # Draw ellipse around and label target object
          if target_list[current_frame] is not None: # This happens as long as there is at least one detected object on screen
            object_ID, b, target_conf = target_list[current_frame]
            centroid = calc_centroid(b)
            centroid = (int(scale * centroid[0]) + left_padding, int(scale * centroid[1]) + top_padding)
            # cv2.putText(frame, object_ID, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             (0, 255, 0), 2)
            horz_rad = int(scale * (b[2] - b[0])/2)
            vert_rad = int(scale * (b[3] - b[1])/2)
            target_color = (0, 255, 0) # bright green
            cv2.ellipse(frame, centroid, (horz_rad, vert_rad), 0, 0, 360, color = target_color, thickness = 2)
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
today = str(datetime.now().month) + "-" + str(datetime.now().day) + "-" + str(datetime.now().year)
time_now = str(datetime.now().hour) + "_" + str(datetime.now().minute)
file_name = participant_id + "_stimulus_" + today + "_" + time_now + '.csv'
print('Outputting stimulus data to ' + file_name)
with open(file_name, 'w') as outfile:

  writer = csv.writer(outfile, delimiter = ',')
  title = ["Participant ID: " + participant_id, "Date: " + today, "Time: " + time_now]
  heading = ["ComputerClock_Timestamp", "Video_Index", "Object_Detection_Threshold", "Target_Name", "Target_Confidence", "TargetX", "TargetY", "TargetXRadius", "TargetYRadius"]
  writer.writerow(title)
  writer.writerow(heading)

  # Create fullscreen video display window
  black_screen = np.zeros((screen_width, screen_height, 3))
  cv2.line(black_screen,
           (int(screen_height/2)-10, int(screen_width/2)),
           (int(screen_height/2)+10, int(screen_width/2)),
           (255, 255, 255), 3, cv2.LINE_AA)
  cv2.line(black_screen,
           (int(screen_height/2), int(screen_width/2)-20),
           (int(screen_height/2), int(screen_width/2)+20),
           (255, 255, 255), 2, cv2.LINE_AA)
  cv2.namedWindow('Video Frame', cv2.WND_PROP_FULLSCREEN)
  cv2.setWindowProperty('Video Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
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
    output = smooth_and_display_objects(video_idx=video_idx,
                                        confidence_threshold=confidence_threshold)

    # Display black screen for 5 seconds between each video
    cv2.imshow('Video Frame', black_screen)
    cv2.waitKey(5000)

    # Write trial output
    for row in output:
      writer.writerow(row)

  cv2.destroyAllWindows()
