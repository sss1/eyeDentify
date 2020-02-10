"""This module specifies the Participant class."""

from typing import List

import experiment_video

class Participant:
  """All data specific to a single participant's experiment.
  """
  def __init__(self, ID: int, videos: List[experiment_video.ExperimentVideo]):
    """
    Args:
      ID: Participant ID
      videos: Sequence of participant data for each video, sorted by video index
    """
    self.ID = ID
    self.videos = videos
