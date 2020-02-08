"""This module specifies the frame class."""

import object_frame

class ExperimentFrame:
  """Eye-tracking and target information for a single frame."""
  def __init__(self, target: object_frame.ObjectFrame):
    """Initialize with target information; eyetrack is added later.
    
    Args:
      target: Target object during that frame.
    """
    self.target = target
    self.t = target.t
    self.video_idx = target.video_idx
    self.video_frame = target.video_frame

  def set_eyetrack(self, gazeX, gazeY, diam):
    self.gaze = (gazeX, gazeY)
    self.diam = diam

