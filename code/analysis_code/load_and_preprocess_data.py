import collections
import csv, sys
import numpy as np

from typing import List, Tuple

import util
import experiment_frame
import experiment_video
import object_frame
import participant

experiment_data_dir = '../../data/experiment1/'
VIDEOS = range(1, 15)
Y_CORRECTION = -60

def align_display_to_video(gaze_x: float, gaze_y: float) -> Tuple[float, float]:
  """Aligns coordinates of displayed video/gaze to the raw video coordinates.
  
  Due to the Windows top menu bar, the displayed video is 60 pixels lower than
  its nominal coordinates. To align the experiment with nominal positions of
  objects in the video, we subtract 60 from the experiment y-coordinates.
  """
  return gaze_x, gaze_y + Y_CORRECTION

def load_eyetrack(participantID : int) -> np.ndarray:
  fname = experiment_data_dir + str(participantID).zfill(2) + '_eyetracking.csv'
  with open(fname, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    eyetrack = []
    EyetrackRow = collections.namedtuple('EyetrackRow', next(reader))
    for row in map(EyetrackRow._make, reader):
      timestamp = float(row.ComputerClock_Timestamp)
      gaze_x, gaze_y = align_display_to_video(
          get_best(float(row.LeftEye_GazeX), float(row.RightEye_GazeX)),
          get_best(float(row.LeftEye_GazeY), float(row.RightEye_GazeY)))
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
    reader = csv.reader(f, delimiter=',')
    frames = []
    current_video = None
    next(reader) # Skip experiment metadata row
    StimulusRow = collections.namedtuple('StimulusRow', next(reader))
    for row in map(StimulusRow._make, reader):
      video_idx = int(row.Video_Index)
      if video_idx != current_video:
        video_frame = 0
        current_video = video_idx
      [target_class_name, target_object_index] = row.Target_Name.split('_')
      t = float(row.ComputerClock_Timestamp)
      target_centroid = align_display_to_video(float(row.TargetX),
                                               float(row.TargetY))
      target_size = (float(row.TargetXRadius), float(row.TargetYRadius))
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
    raise ValueError('Eye-tracking starts after stimulus.')
  if eyetrack[-1, 0] <= frames[-1].t:
    raise ValueError('Eye-tracking ends before stimulus.')
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
  print('# of NaNs before interpolation: {}'.format(np.count_nonzero(
      np.isnan(eyetrack))))
  frames = load_stimulus(participantID)
  util.impute_missing_data_D(eyetrack, max_len = 10)
  print('# of NaNs after  interpolation: {}\n'.format(np.count_nonzero(
      np.isnan(eyetrack))))
  synchronize_eyetracking_with_stimulus(eyetrack, frames)

  videos = []
  for video_idx in VIDEOS:
    videos.append(experiment_video.ExperimentVideo(
        video_idx, [f for f in frames if f.video_idx == video_idx]))

  return participant.Participant(participantID, videos)
