# This module performs the analyses for Experiment 1: Guided Viewing with Detected Targets
import pickle, sys

import hmm
import load_and_preprocess_data
import util

SIGMA = 1
TAU = 0.9

VIDEOS = range(1, 15)
NUM_PARTICIPANTS = 1

DETECTION_DATA_DIR = '../../data/detected_objects'

# Load participant data
participants = [load_and_preprocess_data.load_participant(i) for i in range(NUM_PARTICIPANTS)]
print('Loaded data from {} participants.'.format(NUM_PARTICIPANTS))

# Load object detection data
detected_objects = []
for video_idx in VIDEOS:
  detection_data_fname = '{}/{}.pickle'.format(DETECTION_DATA_DIR,
                                               str(video_idx).zfill(2))
  print('Loading object detection data from {}...'.format(detection_data_fname))
  with open(detection_data_fname, 'rb') as in_file:
    all_frames = pickle.load(in_file)
  detected_objects.append(util.smooth_objects(all_frames))

for participant in participants:
  for (experiment_video_data, video_objects) in zip(participant.videos, detected_objects):
    print('Initializing HMM...')
    trial_hmm = hmm.HMM(SIGMA, TAU)
    print('Running HMM forwards pass...')
    for (frame_idx, (experiment_frame_data, detected_objects_in_frame)) in enumerate(zip(experiment_video_data.frames, video_objects)):
      trial_hmm.forwards_update(experiment_frame_data.gaze,
                          detected_objects_in_frame)
    print('Running HMM backwards pass...')
    # TODO: run trial_hmm.backwards() to retrieve MLE, and compare with target
