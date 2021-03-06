"""This module implements the HMM class, which fits an HMM to a data sequence.
"""

from collections import namedtuple
import numpy
import math
from typing import Dict, List, NewType, Tuple

from classes.object_frame import ObjectFrame

# A single cell in the dynamic programming table
Cell = namedtuple('Cell', ['partial_max_log_likelihood', 'predecessor'])

FrameTable = NewType('FrameTable', Dict[ObjectFrame, Cell])

def max_likelihood_key(frame_table: FrameTable):
    return lambda obj: frame_table[obj].partial_max_log_likelihood

class _HMM:
  """An hidden Markov model of a single data sequence.
  
  Example usage:
    hmm = HMM(sigma, tau)
    for experiment_frame, objects_in_frame in zip(experiment_frames, detected_objects):
      hmm.forwards_update(experiment_frame.gaze, objects_in_frame)
    mle = hmm.backwards()

  Hidden Attributes:
    log_likelihood_table: (List[FrameTable]) for each frame, a dict mapping each
      object to its partial maximum log-likelihood and most likely predecessor
  """

  def __init__(self, sigma: float, tau: float):
    """
    Args:
      tau: Nominal probability that the participant stays on the same object
        between two consecutive frames.
      sigma: Scaling factor of HMM emission distribution
    """
    self.sigma = sigma
    self.tau = tau
    self.log_likelihood_table = []

  def forwards_update(self, gaze: Tuple[float, float],
                      objects_in_frame: List[ObjectFrame]):
    """Performs an update step of the forwards algorithm based on input data.

    Args:
      experiment_frame: a single frame of participant data
      objects_in_frame: list of objects detected in frame
    """

    if not self.log_likelihood_table:
      # This is the first frame; only use emission probabilities
      new_frame_table = {}
      for obj in objects_in_frame:
        new_frame_table[obj] = Cell(obj.log_emission_density(gaze, self.sigma),
                                    None)
    else:
      new_frame_table = self._compute_next_frame_table(
          self.log_likelihood_table[-1], gaze, objects_in_frame)
    self.log_likelihood_table.append(new_frame_table)

  def _compute_next_frame_table(
          self, prev_frame_table: FrameTable, gaze: Tuple[float, float],
          objects_in_frame: List[ObjectFrame]):
    """Computes a frame_table using a previous frame table.

    Args:
      prev_frame_table: log-likelihood table from previous frame
      gaze: (x, y) coordinates of gaze

    NOTE: Depending on sigma and tau, this implementation may bias transitions
    to frames where the tracked object disappears.
    """
    num_new_objects = len(objects_in_frame)
    if math.isnan(gaze[0]) or math.isnan(gaze[1]):
      return {None : Cell(0.0, None)}
    new_frame_table = {obj : Cell(float('-inf'), None)
                       for obj in objects_in_frame}

    for prev_obj in prev_frame_table:

      prev_obj_partial_log_likelihood = \
              prev_frame_table[prev_obj].partial_max_log_likelihood
      prev_obj_in_new_frame = (prev_obj in objects_in_frame)

      for new_obj in objects_in_frame:

        if prev_obj_in_new_frame and prev_obj == new_obj:
          transition_probability = self.tau
        elif prev_obj_in_new_frame:
          # This case only occurs if there is >1 object, so we don't divide by 0
          transition_probability = (1 - self.tau)/(num_new_objects - 1)
        else:
          transition_probability = 1/num_new_objects

        new_partial_log_likelihood = (
            prev_obj_partial_log_likelihood
            + math.log(transition_probability)
            + new_obj.log_emission_density(gaze, self.sigma))

        if (new_partial_log_likelihood
            > new_frame_table[new_obj].partial_max_log_likelihood):
          new_frame_table[new_obj] = Cell(new_partial_log_likelihood, prev_obj)
    return new_frame_table

  def backwards(self) -> List[ObjectFrame]:
    """Runs the backwards algorithm to compute the object sequence MLE.

    Returns:
      Maximum likelihood object sequence
    """

    mle_backwards = []

    # Get most likely final state
    frame_table = self.log_likelihood_table[-1]
    current = max(frame_table, key=max_likelihood_key(frame_table),
                  default=None)

    restart = False
    for frame_table in self.log_likelihood_table[::-1]:
      mle_backwards.append(current)
      if restart:
        current = max(frame_table, key=max_likelihood_key(frame_table),
                      default=None)
      else:
        current = frame_table[current].predecessor

      if current is None:
        restart = True

    return mle_backwards[::-1]

def forwards_backwards(sigma, tau, experiment_video, video_objects):
    trial_hmm = _HMM(sigma, tau)
    for (experiment_frame_data, detected_objects_in_frame) \
        in zip(experiment_video.frames, video_objects):
      trial_hmm.forwards_update(experiment_frame_data.gaze,
                          detected_objects_in_frame)
    return trial_hmm.backwards()
