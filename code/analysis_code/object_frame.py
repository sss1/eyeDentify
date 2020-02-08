"""This module specifies the ObjectFrame class."""

import scipy.stats

class ObjectFrame:
  """Information about a single object in a single frame."""
  def __init__(self, t: float, video_idx: int, video_frame: int,
          object_name: str, x: float, y: float, x_radius: float,
          y_radius: float):
    """Initialize an ObjectFrame object.

    Args:
      t: Time (in ms) since the epoch, according to the experiment computer.
      video_idx: Index (between 1-14, inclusive) of the video being displayed
      video_frame: Index of the frame of the video being displayed
      target_name: Name (COCO class) of the target object
      x: x-coordinate of center of object bounding box
      y: y-coordinate of center of object bounding box
      x_radius: half-width of the target
      y_radius: half-height of the target
    """
    self.t = t
    self.video_idx = video_idx
    self.video_frame = video_frame

    self.class_name = object_name.split('_')[0]
    self.object_idx = int(object_name.split('_')[1])
    self.centroid = (x, y)
    self.size = (x_radius, y_radius)

  def emission_density(self, point):
    """Returns the value of the object's emission density at a point."""
    mu = np.array([self.x, self.y])
    Sigma = (sigma**2) * np.array([[self.x_radius**2, 0],
                                 [0, self.y_radius**2]])

    return stats.multivariate_normal.logpdf(eyetrack, mean=mu, cov=Sigma)
