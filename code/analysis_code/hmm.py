"""This module fits an HMM to a sequence of eye-tracking and stimulus data.
"""

import numpy

def fit_HMM(eyetrack_seq : list[tuple[float, float]],
        stimulus_seq,
        sigma : float,
        stay_prob : float):
""" Use MLE to fit HMM hidden state sequence to observed eyetrack/stimulus data

Inputs:
    eyetrack_seq: N-by-2 array of floats, encoding eyetracking coordinates in
      each frame
    stimulus_seq: List of stimulus frames. Each frame contains a list of objects
      present in that frame.
    sigma: Spatial scaling factor of emission distribution. The standard
      deviations of the x- and y-marginals of Gaussian around an object is
      sigma times the respective dimension of the object
    stay_prob: Probability of staying on the same object between two consecutive
      frames; on a frame with K objects, each switch probability is:
                (1 - stay_prob)/K
"""

  N = eyetrack.shape[0] # number of frames
  state_seq = np.zeros(N, dtype=int)
  # For each object in each frame, record maximum log-likelihood of ending on
  # that frame, based on data from previous frames, as well as the preceding
  # object that maximizes that likelihood
  likelihood_seq = []
  predecessor_seq = []

  # TODO: Initialize likelihood_seq

  for (eyetrack_frame, stim_frame) in zip(eyetrack_seq, stimulus_seq):
    marginal_likelihoods = {obj : frame_likelihood(eyetrack_frame, obj, sigma)
            for obj in stim_frame}
    # TODO: iterate through objects from previous frame and compute
