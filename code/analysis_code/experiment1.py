# This module performs the analyses for Experiment 1: Guided Viewing with Detected Targets

# Data is loaded and stored as an array of "subjects", each of which is an array of trials

num_subjects = 1
data_dir = '../../data/experiment1/'

class TrialData:
  def __init__(self, video_idx, eyetrack, stimulus):
    self.video_idx = video_idx
    self.eyetrack = eyetrack
    self.stimulus = stimulus

def load_subject(subjectID):
  subject = []
  # TODO: load subject trials
  return subject
