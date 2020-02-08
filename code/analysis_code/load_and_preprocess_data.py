import csv, sys
# import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(1, '../util')
import util
import experiment_frame
import object_frame
import object_trajectory

experiment_data_dir = '../../data/experiment1/'

Y_CORRECTION = -60

def align_gaze_to_stimulus(gaze_x, gaze_y):
  """Due to the top menu bar, the video is displayed 60 pixels lower than its
  nominal coordinates. To align the eyetracking and the video, we subtract 60
  from the eyetracking y-coordinates.
  """
  return gaze_x, gaze_y + Y_CORRECTION

class Participant:
  def __init__(self, ID, frames_by_video):
    self.ID = ID
    self.frames_by_video = frames_by_video

def load_eyetrack(participantID : int):
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
        gaze_x, gaze_y = align_gaze_to_stimulus(get_best(row[3], row[5]),
                                                get_best(row[4], row[6]))
        diam = get_best(row[7], row[8])
        eyetrack.append([timestamp, gaze_x, gaze_y, diam])
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
        targetX = float(row[3])
        targetY = float(row[4]) + Y_CORRECTION
        targetXRadius = float(row[5])
        targetYRadius = float(row[6])
        target = object_frame.ObjectFrame(t, video_idx, video_frame,
                                          target_name, targetX, targetY,
                                          targetXRadius, targetYRadius)
        frames.append(experiment_frame.ExperimentFrame(target))
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
  print('# of NaNs before interpolation: {}'.format(np.count_nonzero(np.isnan(eyetrack))))
  frames = load_stimulus(participantID)
  util.impute_missing_data_D(eyetrack, max_len = 10)
  print('# of NaNs after  interpolation: {}\n'.format(np.count_nonzero(np.isnan(eyetrack))))
  synchronize_eyetracking_with_stimulus(eyetrack, frames)
  frames_by_video = [[f for f in frames if f.video_idx == video_idx] for video_idx in range(1, 15)]

  return Participant(participantID, frames_by_video)
