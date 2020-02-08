import numpy as np
import math
import sys

import centroidtracker

# Given a sequence X of D-dimensional vectors, performs __impute_missing_data (independently) on each dimension of X
# X is N X D, where D is the dimensionality and N is the sample length
def impute_missing_data_D(X, max_len = 10):
  D = X.shape[1]
  for d in range(D):
    X[:, d] = __impute_missing_data(X[:, d], max_len)
  return X

# Given a sequence X of floats, replaces short streches (up to length max_len) of NaNs with linear interpolation
# For example, if
# X = np.array([1, NaN, NaN,  4, NaN,  6])
# then
# impute_missing_data(X, max_len = 1) == np.array([1, NaN, NaN, 4, 5, 6])
# and
# impute_missing_data(X, max_len = 2) == np.array([1, 2, 3, 4, 5, 6])
def __impute_missing_data(X, max_len):
  last_valid_idx = -1
  for n in range(len(X)):
    if not math.isnan(X[n]):
      if last_valid_idx < n - 1: # there is missing data and we have seen at least one valid eyetracking sample
        if n - (max_len + 1) <= last_valid_idx: # amount of missing data is at most than max_len
          if last_valid_idx == -1: # No previous valid data (i.e., first timepoint is missing)
            X[0:n] = X[n] # Just propogate first valid data point
          else:
            first_last = np.array([X[last_valid_idx], X[n]]) # initial and final values from which to linearly interpolate
            new_len = n - last_valid_idx + 1
            X[last_valid_idx:(n + 1)] = np.interp([float(x)/(new_len - 1) for x in range(new_len)], [0, 1], first_last)
      last_valid_idx = n
    elif n == len(X) - 1: # if n is the last index of X and X[n] is NaN
      if n - (max_len + 1) <= last_valid_idx: # amount of missing data is at most than max_len
        X[last_valid_idx:] = X[last_valid_idx]
  return X

# Given a list (over frames) of objects detected by the object detector in each frame,
# Stitches them together into object tracking data
def smooth_objects(all_frames):
  tracker_list = [] 
  # Since we assume that objects cannot change types, we separately run the
  # object tracking algorithm for each object type
  obj_classes = [obj['name'] for frame in all_frames for obj in frame]
  # print('Unique object types: ' + str(set(obj_classes)))
  obj_class_counts = [(name, obj_classes.count(name)) for name in set(obj_classes)]
  print('Object class counts: ' + str(obj_class_counts))

  # Initialize a centroid tracker for each object type
  trackers = {obj_type : centroidtracker.CentroidTracker(maxDisappeared = 15)
              for obj_type in set(obj_classes)}

  for (frame_idx, frame) in enumerate(all_frames):

    new_frame_list = []
    for obj_type in set(obj_classes):
      trackers[obj_type].update([obj['box_points'] for obj in frame if obj['name'] is obj_type]) # Update the centroid tracker

      for (ID, centroid) in trackers[obj_type].objects.items():
        new_ID = obj_type + '_' + str(ID) # Concatenate object type with object ID
        for obj in frame:
          if obj['name'] is obj_type:
          # Since we need to output (ID, bounding box) and Centroid Tracker doesn't record bounding boxes
          # match each ID with its bounding box by centroid; this solution implicitly assumes each object of a
          # specified type within each frame has a distinct centroid
            obj_centroid = centroidtracker.calc_centroid(obj['box_points'])
            if max(abs(obj_centroid[0] - centroid[0]), abs(obj_centroid[1] - centroid[1])) < sys.float_info.epsilon:
              new_frame_list.append((new_ID, obj['box_points']))

    tracker_list.append(new_frame_list)

  return tracker_list

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
