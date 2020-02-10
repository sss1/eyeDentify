"""This module specifies the ObjectFrame class."""

import scipy.stats

from typing import Tuple

class ObjectFrame:
  """Information about a single object in a single frame."""
  def __init__(self, class_name: str, object_index: int,
               centroid: Tuple[float, float],
               size: Tuple[float, float]):
    """Initialize an ObjectFrame object.

    Args:
      class_name: COCO class of the object
      object_index: Index (in order of appearance) of that object within its
        object class and video (e.g., chair 5 is the 5th chair appearing in the
        video). Note that the (class_name, object_index) pair uniquely
        identifies a object (which may last multiple frames) within a video
      centroid: (x, y)-coordinates of center of object bounding box
      size: (half-width, half-height) of object bounding box
    """
    self.class_name = class_name
    self.object_index = object_index
    self.centroid = centroid
    self.size = size

  def is_same_object(self, other):
    return (self.class_name == other.class_name
            and self.object_index == other.object_index)

  def emission_density(self, point):
    """Returns the value of the object's emission density at a point."""
    mu = np.array([self.x, self.y])
    Sigma = (sigma**2) * np.array([[self.x_radius**2, 0],
                                 [0, self.y_radius**2]])

    return stats.multivariate_normal.logpdf(eyetrack, mean=mu, cov=Sigma)
