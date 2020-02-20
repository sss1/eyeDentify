"""This module specifies the ExperimentFrame class."""

import math

import object_frame

class ExperimentFrame:
  """Eye-tracking and target information for a single frame."""
  def __init__(self, video_idx: int, t: float, video_frame: int,
               target: object_frame.ObjectFrame):
    """Initialize with target information; eyetrack is added later.
    
    Args:
      video_idx: Index (between 1-14, inclusive) of the video being displayed.
      t: Time (in ms) since the epoch, according to the experiment computer.
      video_frame: Index of the frame of the video being displayed
      target: Target object during that frame.
    """
    self.video_idx = video_idx
    self.t = t
    self.target = target
    self.video_frame = video_frame

  def set_eyetrack(self, gaze_x, gaze_y, diam):
    self.gaze = (gaze_x, gaze_y)
    self.gaze_is_missing = math.isnan(gaze_x) or math.isnan(gaze_y)
    self.diam = diam

  def set_detected_objects(self, detected_objects):
    self.detected_objects = detected_objects
