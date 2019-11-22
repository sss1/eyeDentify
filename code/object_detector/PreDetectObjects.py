"""This module precomputes and saves the object detections for each video."""

import os
import pickle

from ObjectDetector import detect_objects

data_dir_in = '../../data/MOT17_videos'
data_dir_out = '../../data/detected_objects'

num_videos = 14
detection_confidence_thresholds = [30, 60, 90]
for video_idx in range(1, num_videos + 1):

  input_file_path = \
    '{dir}/{video_idx}.mp4'.format(dir=data_dir_in,
                                   video_idx=str(video_idx).zfill(2))

  print('Processing video ' + input_file_path + '...')
  for confidence_threshold in detection_confidence_thresholds:
    print('at confidence threshold ' + str(confidence_threshold) + '...')

    # This line runs the object detector
    detected_objects = detect_objects(input_file_path, confidence_threshold)

    # Save results in pickle file
    output_path = \
      '{dir}/video{video_idx}_threshold{conf}.pickle'.format(dir=data_dir_out,
              video_idx=str(video_idx).zfill(2),
              conf=confidence_threshold)
    with open(output_path, 'wb') as out_file:
      pickle.dump(detected_objects, out_file, protocol=pickle.HIGHEST_PROTOCOL)
    print('Output results to ' + output_path)
