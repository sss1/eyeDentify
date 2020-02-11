"""This module implements the HMM class, which fits an HMM to a data sequence.
"""

import numpy

from typing import Dict, List

import experiment_frame
import object_frame

class HMM:
  """An hidden Markov model of a single data sequence.
  
  Example usage:
    hmm = HMM(sigma, tau)
    for experiment_frame, objects_in_frame in zip(experiment_frames, detected_objects):
      hmm.update(experiment_frame, objects_in_frame)
    mle = hmm.backwards()

  Hidden Attributes:
    likelihood_table: mapping from each object to its partial maximum likelihood
      and most likely predecessor
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
    self.likelihood_table : Dict[object_frame.ObjectFrame,
                                 (float, object_frame.ObjectFrame)] = {}

  def forwards_update(experiment_frame: experiment_frame.ExperimentFrame,
                      objects_in_frame: List[object_frame.ObjectFrame]):
    """Performs an update step of the forwards algorithm based on input data.
    Args:
      experiment_frame: a single frame of participant data
      objects_in_frame: list of objects detected in frame
    """
    raise NotImplemented()

  def backwards() -> List[object_frame.ObjectFrame]:
    """Runs the backwards algorithm to compute the object sequence MLE.

    Returns:
      Maximum likelihood object sequence
    """
    raise NotImplemented()
