import collections
import csv, sys
import numpy as np

from typing import List, Tuple

import util
import classes.experiment_video as experiment_video
import classes.participant as participant
import classes.experiment_frame as experiment_frame
import classes.object_frame as object_frame

experiment_data_dir = '../../data/experiment1/'
VIDEOS = range(1, 15)

def load_eyetrack(participantID : int) -> np.ndarray:
  fname = experiment_data_dir + str(participantID).zfill(2) + '_eyetracking.csv'
  with open(fname, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    eyetrack = []
    EyetrackRow = collections.namedtuple('EyetrackRow', next(reader))
    for row in map(EyetrackRow._make, reader):
      timestamp = float(row.ComputerClock_Timestamp)
      gaze_x = get_best(float(row.LeftEye_GazeX), float(row.RightEye_GazeX))
      gaze_y = get_best(float(row.LeftEye_GazeY), float(row.RightEye_GazeY))
      diam = get_best(float(row.LeftEye_Diam), float(row.RightEye_Diam))
      eyetrack.append([timestamp, gaze_x, gaze_y, diam])
  print('Loading {} rows of eyetracking data from {}.'.format(len(eyetrack),
                                                              fname))
  return np.array(eyetrack)

def get_best(left: float, right: float):
  """For gaze and diameter we get separate left and right eye measurements.
     We recode missing values from 0.0 to NaN. If one eye's data is missing,
     take the other eye's data; else, take the average.
  """
  if left < sys.float_info.epsilon and right < sys.float_info.epsilon:
    return float('nan')
  if left < sys.float_info.epsilon:
    return right
  if right < sys.float_info.epsilon:
    return left
  return (left + right)/2

def load_stimulus(participantID: int) -> List[experiment_frame.ExperimentFrame]:
  fname = experiment_data_dir + str(participantID).zfill(2) + '_stimulus.csv'
  with open(fname, 'r') as f:

    frames = []
    current_video = None

    reader = csv.reader(f, delimiter=',')
    next(reader) # Skip experiment metadata row
    StimulusRow = collections.namedtuple('StimulusRow', next(reader))

    for row in map(StimulusRow._make, reader):
      video_idx = int(row.Video_Index)
      if video_idx != current_video:
        video_frame = 0
        current_video = video_idx
      target_class_name = row.Target_Name.split('_')[0]
      target_object_index = int(row.Target_Name.split('_')[1])
      t = float(row.ComputerClock_Timestamp)
      if t < 1e12:
        # Due to a bug, some stimulus were recorded in seconds rather than ms
        t *= 1000
      target_centroid = (int(row.TargetX), int(row.TargetY))
      target_size = (int(row.TargetXRadius), int(row.TargetYRadius))
      target = object_frame.ObjectFrame(target_class_name,
                                        target_object_index,
                                        target_centroid,
                                        target_size)
      frames.append(experiment_frame.ExperimentFrame(video_idx, t, video_frame,
                                                     target))
      video_frame += 1
  return frames

def synchronize_eyetracking_with_stimulus(eyetrack, frames):
  """Interpolates eyetracking frames to same timepoints as stimulus frames."""
  eyetrack_idx = 0
  if eyetrack[0, 0] >= frames[0].t:
    error = (eyetrack[0, 0] - frames[0].t)/1000
    print('EYE-TRACKING STARTS {} SECONDS AFTER STIMULUS.'.format(error))
    eyetrack[0] = np.array([frames[0].t - 1, float('nan'), float('nan'), float('nan')])
  if eyetrack[-1, 0] <= frames[-1].t:
    error = (frames[-1].t - eyetrack[-1, 0])/1000
    print('EYE-TRACKING ENDS {} SECONDS BEFORE STIMULUS.'.format(error))
    eyetrack[-1] = np.array([frames[-1].t + 1, float('nan'), float('nan'), float('nan')])
  for frame in frames:
    while eyetrack[eyetrack_idx, 0] < frame.t:
      eyetrack_idx += 1
    t0, t1 = eyetrack[(eyetrack_idx - 1):(eyetrack_idx + 1), 0]
    # At this point, t0 <= frame.t < t1. Linearly interpolate x and y based on
    # the surrounding x0, x1, y0, and y1.
    theta = (frame.t - t0)/(t1 - t0)
    gaze_x, gaze_y, diam = ((1 - theta) * eyetrack[eyetrack_idx - 1, 1:]
                          +   theta   * eyetrack[eyetrack_idx, 1:])
    frame.set_eyetrack(gaze_x, gaze_y, diam)

def load_participant(participantID: int) -> participant.Participant:

  eyetrack = load_eyetrack(participantID)
  frames = load_stimulus(participantID)
  util.impute_missing_data_D(eyetrack, max_len = 10)
  synchronize_eyetracking_with_stimulus(eyetrack, frames)

  videos = []
  for video_idx in VIDEOS:
    videos.append(experiment_video.ExperimentVideo(
        video_idx, [f for f in frames if f.video_idx == video_idx]))

  return participant.Participant(participantID, videos)
