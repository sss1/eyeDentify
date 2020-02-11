# This module performs the analyses for Experiment 1: Guided Viewing with Detected Targets
# import numpy as np
import pickle, sys

import load_and_preprocess_data
import util
import centroidtracker

from typing import List

import participant

VIDEOS = range(1, 15)
NUM_PARTICIPANTS = 1

# Load participant data
participants = [load_and_preprocess_data.load_participant(i) for i in range(NUM_PARTICIPANTS)]

print('Loaded data from {} participants.'.format(NUM_PARTICIPANTS))

# Load object detection data
detected_objects = []
for video_idx in VIDEOS:
  with open('../../data/detected_objects/' + str(video_idx).zfill(2) + '.pickle', 'rb') as in_file:
    all_frames = pickle.load(in_file)
  detected_objects.append(util.smooth_objects(all_frames))

for participant in participants:
  for (experiment_video_data, video_objects) in zip(participant.videos, detected_objects):
    # TODO: Initialize HMM here
    for (experiment_frame_data, detected_objects_in_frame) in zip(experiment_video_data.frames, video_objects):
      print(experiment_frame_data.gaze, experiment_frame_data.target.centroid, experiment_frame_data.target.class_name)
      print([obj.centroid for obj in detected_objects_in_frame])
      print()
      # TODO: Update HMM here

# for participant in participants:
  # TODO:
  # 1) Load object detection data
  # 2) Run HMM
  # 3) Omit data for 300 ms after each target switch
  # 4) Compute agreement between target and HMM
