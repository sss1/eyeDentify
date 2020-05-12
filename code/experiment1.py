"""This module performs analyses for Experiment 1: Guided Viewing with Detected Targets."""
import numpy as np
import pickle

import hmm
from load_and_preprocess_data import load_participant
import metrics
import util

# Preprocessing parameters
MAX_MISSING_PROPORTION = 0.25

# HMM hyperparameters
SIGMA = 1
TAU = 0.9

VIDEOS = range(1, 15)
PARTICIPANTS = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
]

print('Parameters:')
print('SIGMA: {}\nTAU: {}\nVIDEOS: {}\nPARTICIPANTS: {}'
      .format(SIGMA, TAU, VIDEOS, PARTICIPANTS))

DETECTION_DATA_DIR = '../data/detected_objects'

# Load participant data
participants = [load_participant(i) for i in PARTICIPANTS]
print('Loaded data from {} participants.'.format(len(PARTICIPANTS)))

# Discard participants with too much missing data
participants = [participant for participant in participants
                if participant.mean_proportion_missing < MAX_MISSING_PROPORTION]

print('Keeping {} participants: {}'
      .format(len(participants), [p.ID for p in participants]))

# Load object detection data
detected_objects = []
for video_idx in VIDEOS:
  detection_data_fname = '{}/{}.pickle'.format(DETECTION_DATA_DIR,
                                               str(video_idx).zfill(2))
  print('Loading object detection data from {}...'.format(detection_data_fname))
  with open(detection_data_fname, 'rb') as in_file:
    all_frames = pickle.load(in_file)
  detected_video_objects = util.smooth_objects(all_frames)
  util.align_objects_to_screen(video_idx, detected_video_objects)
  detected_objects.append(detected_video_objects)

participant_accuracies = []
for participant in participants:
  print('Running participant {}...'.format(participant.ID))
  participant_videos = [participant.videos[i-1] for i in VIDEOS]
  video_accuracies = []
  for (experiment_video, video_objects) \
          in zip(participant_videos, detected_objects):

    mle = hmm.forwards_backwards(SIGMA, TAU, experiment_video, video_objects)
    ground_truth = [frame.target for frame in experiment_video.frames]
    video_accuracy = metrics.compute_accuracy(mle, ground_truth)
    print('Video {} accuracy: {}'.format(experiment_video.video_idx, video_accuracy))
    video_accuracies.append(video_accuracy)

  participant_accuracy_mean, participant_accuracy_ste = metrics.mean_and_ste(
          video_accuracies)
  print('Participant accuracy: {} +/- {}'.format(participant_accuracy_mean,
                                                 participant_accuracy_ste))
  participant_accuracies.append(participant_accuracy_mean)

accuracy_mean, accuracy_ste = metrics.mean_and_ste(participant_accuracies)
print('Overall accuracy: {} +/- {}'.format(accuracy_mean, accuracy_ste))
