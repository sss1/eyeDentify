"""This module specifies the Participant class."""

import numpy as np
from typing import List

from classes.experiment_video import ExperimentVideo

class Participant:
  """All data specific to a single participant's experiment.
  """
  def __init__(self, ID: int, videos: List[ExperimentVideo]):
    """
    Args:
      ID: Participant ID
      videos: Sequence of participant data for each video, sorted by video index
    """
    self.ID = ID
    self.videos = videos
    self.mean_proportion_missing = np.mean(
        [video.proportion_missing for video in videos])
