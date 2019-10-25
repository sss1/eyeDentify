import csv, sys
# import numpy as np

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
    self.has_gaze = gazeX > 0.0 and gazeY > 0.0
    if self.has_gaze:
      self.gaze = (gazeX, gazeY)
    else:
      self.gaze = (float('nan'), float('nan'))

    self.has_diam = diam > 0.0
    if self.has_diam:
      self.diam = diam
    else:
      self.diam = float('nan')

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
  return eyetrack#np.array(eyetrack)

# For GazeX, GazeY, and Diam, we get separate left and right eye measurements.
# Missing values are encoded as 0.0. If either eye's data is missing, take the
# other eye's data; else, take the average.
def get_best(left, right):
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

def interpolate_missing_eyetracking_data(eyetrack):
  # TODO: Interpolate missing eye-tracking data
  raise NotImplementedError

def synchronize_eyetracking_with_stimulus(eyetrack, frames):
  raise NotImplementedError
  for (eyetrack_frame, frame) in zip(eyetrack, frames):
  # TODO: Re-align eye-tracking data to stimulus frames by linearly interpolating
  # gaze and diam from surrounding eyetracking frames
    frame.set_eyetrack(*eyetrack_frame[1:])

def load_participant(participantID):
  eyetrack = load_eyetrack(participantID)
  frames = load_stimulus(participantID)
  interpolate_missing_eyetracking_data(eyetrack)
  synchronize_eyetracking_with_stimulus(eyetrack, frames)
  # print('Data from first 5 frames:')
  # for f in frames[:5]:
  #   print(vars(f))
  return Participant(participantID, frames)
