# This module performs the analyses for Experiment 1: Guided Viewing with Detected Targets
# import numpy as np
import pickle

import load_and_preprocess_data

# Load participant data
num_participants = 1
# participants = [load_and_preprocess_data.load_participant(i) for i in range(num_participants)]
# participants = load_participants(num_participants)

# Load object detection data
detected_objects = []
for video_idx in range(1, 15):
  with open('../../data/detected_objects/' + str(video_idx).zfill(2) + '.pickle', 'rb') as in_file:
    detected_objects.append(pickle.load(in_file))

# for participant in participants:
  # TODO:
  # 1) Load object detection data
  # 2) Run HMM
  # 3) Omit data for 300 ms after each target switch
  # 4) Compute agreement between target and HMM
