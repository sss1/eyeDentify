import csv, sys
# import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(1, '../util')
import util

experiment_data_dir = '../../data/experiment1/'

class Participant:
  def __init__(self, ID, frames):
    self.ID = ID
    self.frames = frames

class Frame:
  def __init__(self, t, video_idx, video_frame, target_name, targetX, targetY, targetXRadius, targetYRadius):
    self.t = t # float timestamp at which frame was displayed
    self.video_idx = video_idx # int between 1 and 14 (inclusive) indicating which video is being displayed
    self.video_frame = video_frame # int frame number in video indicated by video_idx
    self.target_class = target_name.split('_')[0] # string indicating class of target object (e.g., 'person' or 'car')
    self.target_idx = int(target_name.split('_')[1]) # int index of the target object within that class and video
    self.target_centroid = (targetX, targetY)
    self.target_size = (targetXRadius, targetYRadius)

  def set_eyetrack(self, gazeX, gazeY, diam):
    self.gaze = (gazeX, gazeY)
    self.diam = diam

def load_eyetrack(participantID):
  fname = experiment_data_dir + str(participantID).zfill(2) + '_eyetracking.csv'
  with open(fname, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    eyetrack = []
    row_num = 0
    for row in reader:
      if row_num > 0: # Skip 1 row of header
        # Eyetracking CSV format is:
        # Timestamp,AvgGazeX,AvgGazeY,LeftGazeX,LeftGazeY,RightGazeX,RightGazeY,LeftDiam,RightDiam
        row = [float(x) for x in row]
        timestamp = row[0]
        gazeX = get_best(row[3], row[5])
        gazeY = get_best(row[4], row[6])
        diam = get_best(row[7], row[8])
        eyetrack.append([timestamp, gazeX, gazeY, diam])
      row_num += 1
  print('Loading ' + str(row_num) + ' eyetracking frames from ' + fname + '.')
  return np.array(eyetrack)

# For GazeX, GazeY, and Diam, we get separate left and right eye measurements.
# Missing values are recoded from 0.0 to NaN. If one eye's data is missing,
# take the other eye's data; else, take the average.
def get_best(left, right):
  if left < sys.float_info.epsilon:
    return float('nan')
  if left < sys.float_info.epsilon:
    return right
  if right < sys.float_info.epsilon:
    return left
  return (left + right)/2

def load_stimulus(participantID):
  fname = experiment_data_dir + str(participantID).zfill(2) + '_stimulus.csv'
  with open(fname, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    frames = []
    row_num = 0
    current_video = None
    video_frame = 0
    for row in reader:
      if row_num > 1: # Skip 2 rows of header
        # Stimulus CSV format is:
        # ComputerClock_Timestamp,Video_Index,Target_Name,TargetX,TargetY,TargetXRadius,TargetYRadius
        t, video_idx, target_name = float(row[0]), int(row[1]), row[2]
        if video_idx != current_video:
          video_frame = 0
          current_video = video_idx
        targetX, targetY, targetXRadius, targetYRadius = float(row[3]), float(row[4]), float(row[5]), float(row[6])
        frames.append(Frame(t, video_idx, video_frame, target_name, targetX, targetY, targetXRadius, targetYRadius))
        video_frame += 1
      row_num += 1
  return frames

def synchronize_eyetracking_with_stimulus(eyetrack, frames):
  eyetrack_idx = 0
  if eyetrack[0, 0] >= frames[0].t:
    raise ValueError('Eye-tracking starts after stimulus.')
  if eyetrack[-1, 0] <= frames[-1].t:
    raise ValueError('Eye-tracking ends before stimulus.')
  for frame in frames:
    while eyetrack[eyetrack_idx, 0] < frame.t:
      eyetrack_idx += 1
    t0, t1 = eyetrack[(eyetrack_idx - 1):(eyetrack_idx + 1), 0]
    # At this point, t0 <= frame.t < t1
    # We linearly interpolate x and y based on the surrounding x0, x1, y0, and y1
    theta = (frame.t - t0)/(t1 - t0)
    gazeX, gazeY, diam = (1 - theta) * eyetrack[eyetrack_idx - 1, 1:] + theta * eyetrack[eyetrack_idx, 1:]
    frame.set_eyetrack(gazeX, gazeY, diam)

def load_participant(participantID):
  eyetrack = load_eyetrack(participantID)
  print('# of NaNs before interpolation: ' + str(np.count_nonzero(np.isnan(eyetrack))))
  frames = load_stimulus(participantID)
  util.impute_missing_data_D(eyetrack, max_len = 10)
  print('# of NaNs after  interpolation: ' + str(np.count_nonzero(np.isnan(eyetrack))))
  synchronize_eyetracking_with_stimulus(eyetrack, frames)
  print('Data from first 5 frames:')
  for f in frames[:5]:
    print(vars(f))
  return Participant(participantID, frames)

print(load_participant(0))
