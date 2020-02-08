"""This module specifies the ObjectTrajectory class."""

from typing import List

import object_frame

class ObjectTrajectory:
  """Information about a single object over an entire video."""
  def __init__(self, frame_seq: List[object_frame.ObjectFrame]):
    this.frame_seq = frame_seq
    this.first_frame = frame_seq[0].video_frame
    this.last_frame = frame_seq[-1].t
