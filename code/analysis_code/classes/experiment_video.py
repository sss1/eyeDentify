"""This module specifies the ExperimentVideo class."""

import numpy as np
from typing import List

from classes.experiment_frame import ExperimentFrame

class ExperimentVideo:
  """Eye-tracking and target information for a single video."""
  def __init__(self, video_idx: int, frames: List[ExperimentFrame]):
    """Initialize with target information; eyetrack is added later.
    
    Args:
      video_idx: Index (between 1-14, inclusive) of the video being displayed.
      frames: Sequence of ExperimentFrames for that video.

    """
    self.video_idx = video_idx
    self.frames = frames
    self.proportion_missing = np.mean(
        [frame.gaze_is_missing for frame in frames])
