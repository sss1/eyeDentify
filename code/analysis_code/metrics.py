"""This module gives metrics for comparing model performance to ground truth."""

from typing import List, Tuple

import math
import numpy as np

# Allow participant 18 frames (300ms) to find new target after switch
GRACE_PERIOD = 18


def compute_accuracy(predicted_seq: List[int], actual_seq: List[int]) -> float:
  predicted_correct = []
  prev_actual = None
  last_switch_idx = 0
  for idx, (predicted, actual) in enumerate(zip(predicted_seq, actual_seq)):
    if actual != prev_actual:
      prev_actual = actual
      last_switch_idx = idx
    elif (idx > last_switch_idx + GRACE_PERIOD
          and predicted != None):
      predicted_correct.append(predicted == actual)
  return np.mean(predicted_correct)

def mean_and_ste(array: List[float]) -> Tuple[float, float]:
  return np.nanmean(array), np.nanstd(array)/math.sqrt(len(array))
