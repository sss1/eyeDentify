import sys
import numpy as np

from centroidtracker import CentroidTracker
from util import calc_centroid

def _smooth_objects(all_frames):
  tracker_list = [] 
  # Since we assume that objects cannot change types, we separately run the
  # object tracking algorithm for each object type
  unique_obj_types = set([obj['name'] for frame in all_frames for obj in frame])
  print('Unique object types: ' + str(unique_obj_types))

  # Initialize a centroid tracker for each object type
  trackers = {obj_type : CentroidTracker(maxDisappeared = 15)
              for obj_type in unique_obj_types}

  for (frame_idx, frame) in enumerate(all_frames):

    new_frame_list = []
    for obj_type in unique_obj_types:
      # Update the centroid tracker
      trackers[obj_type].update([obj['box_points'] for obj in frame
                                 if obj['name'] is obj_type])

      for (ID, centroid) in trackers[obj_type].objects.items():
        new_ID = obj_type + '_' + str(ID) # Concatenate object type with object ID
        for obj in frame:
          if obj['name'] is obj_type:
          # Since we need to output (ID, bounding box) and Centroid Tracker
          # doesn't record bounding boxes match each ID with its bounding box by
          # centroid; this solution implicitly assumes each object of a
          # specified type within each frame has a distinct centroid
            obj_centroid = calc_centroid(obj['box_points'])
            if (abs(obj_centroid[0] - centroid[0]) < sys.float_info.epsilon) and \
                 (abs(obj_centroid[1] - centroid[1]) < sys.float_info.epsilon):
              new_frame_list.append((new_ID, obj['box_points'],
                                     obj['percentage_probability']))

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
    if prev_good_frame < frame_idx - 1:
      # If previous frames were missing
      if target_list[frame_idx] is not None:
        # If current frame is non-missing, we need to interpolate till here

        # Objects should only change on non-missing frames
        target_name = target_list[prev_good_frame][0]
        if target_name == target_list[frame_idx][0]:

          # Get first and last non-missing boxes to interpolate between
          prev_good_box = target_list[prev_good_frame][1]
          current_box = target_list[frame_idx][1]

          num_parts = frame_idx - prev_good_frame
          for frame_to_interpolate in range(prev_good_frame + 1, frame_idx):
            interpolation_idx = frame_to_interpolate - prev_good_frame
            interpolate = (lambda x, y :
                           int(x + interpolation_idx/num_parts * (y - x)))
            interpolated_box = (interpolate(prev_good_box[0], current_box[0]),
                                interpolate(prev_good_box[1], current_box[1]),
                                interpolate(prev_good_box[2], current_box[2]),
                                interpolate(prev_good_box[3], current_box[3]))
            target_list[frame_to_interpolate] = (target_name, interpolated_box,
                                                 float('nan'))

    if target_list[frame_idx] is not None:
      # All frames before (and including) frame_idx have been filled in
      prev_good_frame = frame_idx

  return target_list

def _sample_targets(tracker_list, object_durations, min_duration = 30,
                    mean_duration = 45):
  target_list = []
  next_switch_frame = -1
  for (frame_idx, frame_objects) in enumerate(tracker_list):

    if frame_idx > next_switch_frame:
      # Weight objects by remaining duration to prefer longer-lasting objects
      weights = np.array([object_durations[obj[0]][1] - frame_idx
                          for obj in tracker_list[frame_idx]])
      if np.sum(weights) < sys.float_info.epsilon:
        # No objects in tracker_list
        target_list.append(None)
        continue
      weights = weights / weights.sum()
      current_target_idx = np.random.choice(range(len(weights)), p = weights)
      current_target = tracker_list[frame_idx][current_target_idx][0]
      next_switch_frame = object_durations[current_target][1]
      next_switch_frame = min(next_switch_frame,
                              (frame_idx + min_duration
                               + int(np.random.exponential(mean_duration))))
      attempt = 0
      while current_target not in [obj[0] for obj in tracker_list[next_switch_frame]]:
        attempt += 1
        # if attempt % 100 == 0:
        #   print(attempt)
        if attempt > 1000:
          # For some objects/frames, we may not be able to find a valid
          # next_swith_frame even after many attempts. In this case, simply
          # switch on the very next frame.
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

def generate_target_list(all_frames):
  tracker_list = _smooth_objects(all_frames)
  object_durations = _compute_durations(tracker_list)
  return _sample_targets(tracker_list, object_durations)
