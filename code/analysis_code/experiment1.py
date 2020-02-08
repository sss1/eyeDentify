# This module performs the analyses for Experiment 1: Guided Viewing with Detected Targets
# import numpy as np
import pickle, sys

import load_and_preprocess_data
sys.path.insert(1, '../util')
import util
import centroidtracker

sys.path.insert(1, '../util')
import centroidtracker

# Load participant data
num_participants = 1
participants = [load_and_preprocess_data.load_participant(i) for i in range(num_participants)]

# Load object detection data
detected_objects = []
for video_idx in range(1, 15):
  with open('../../data/detected_objects/' + str(video_idx).zfill(2) + '.pickle', 'rb') as in_file:
    all_frames = pickle.load(in_file)
  detected_objects.append(util.smooth_objects(all_frames))

# TODO: Build ObjectTrajectory objects of each object


for participant in participants:
  for (experiment_video_data, video_objects) in zip(participant.frames_by_video, detected_objects):
    # TODO: Initialize HMM here
    for (experiment_frame_data, detected_objects_in_frame) in zip(experiment_video_data, video_objects):
      print(experiment_frame_data.gaze, experiment_frame_data.target.centroid)
      print([centroidtracker.calc_centroid(obj[1]) for obj in detected_objects_in_frame])
      print()
      # TODO: Update HMM here

# for participant in participants:
  # TODO:
  # 1) Load object detection data
  # 2) Run HMM
  # 3) Omit data for 300 ms after each target switch
  # 4) Compute agreement between target and HMM
