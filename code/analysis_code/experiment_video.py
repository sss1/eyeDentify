"""This module specifies the ExperimentVideo class."""

from typing import List

import experiment_frame

class ExperimentVideo:
  """Eye-tracking and target information for a single video."""
  def __init__(self, video_idx: int,
               frames: List[experiment_frame.ExperimentFrame]):
    """Initialize with target information; eyetrack is added later.
    
    Args:
      video_idx: Index (between 1-14, inclusive) of the video being displayed.
      frames: Sequence of ExperimentFrames for that video.

    """
    self.video_idx = video_idx
    self.frames = frames
