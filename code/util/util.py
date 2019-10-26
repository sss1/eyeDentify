import numpy as np
import math

# Given a sequence X of D-dimensional vectors, performs __impute_missing_data (independently) on each dimension of X
# X is N X D, where D is the dimensionality and N is the sample length
def impute_missing_data_D(X, max_len = 10):
  D = X.shape[1]
  for d in range(D):
    X[:, d] = __impute_missing_data(X[:, d], max_len)
  return X

# Given a sequence X of floats, replaces short streches (up to length max_len) of NaNs with linear interpolation
# For example, if
# X = np.array([1, NaN, NaN,  4, NaN,  6])
# then
# impute_missing_data(X, max_len = 1) == np.array([1, NaN, NaN, 4, 5, 6])
# and
# impute_missing_data(X, max_len = 2) == np.array([1, 2, 3, 4, 5, 6])
def __impute_missing_data(X, max_len):
  last_valid_idx = -1
  for n in range(len(X)):
    if not math.isnan(X[n]):
      if last_valid_idx < n - 1: # there is missing data and we have seen at least one valid eyetracking sample
        if n - (max_len + 1) <= last_valid_idx: # amount of missing data is at most than max_len
          if last_valid_idx == -1: # No previous valid data (i.e., first timepoint is missing)
            X[0:n] = X[n] # Just propogate first valid data point
          else:
            first_last = np.array([X[last_valid_idx], X[n]]) # initial and final values from which to linearly interpolate
            new_len = n - last_valid_idx + 1
            X[last_valid_idx:(n + 1)] = np.interp([float(x)/(new_len - 1) for x in range(new_len)], [0, 1], first_last)
      last_valid_idx = n
    elif n == len(X) - 1: # if n is the last index of X and X[n] is NaN
      if n - (max_len + 1) <= last_valid_idx: # amount of missing data is at most than max_len
        X[last_valid_idx:] = X[last_valid_idx]
  return X
