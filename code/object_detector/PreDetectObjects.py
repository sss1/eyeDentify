"""This module precomputes and saves the object detections for each video."""

import os
import pickle

from ObjectDetector import detect_objects

execution_path = os.getcwd()

num_videos = 14
for video_idx in range(1, num_videos + 1):
  input_file_path = '../../data/MOT17_videos/' + str(video_idx).zfill(2) + '.mp4'
  print('Processing video ' + input_file_path)
  detected_objects = detect_objects(input_file_path)

  # Save results in pickle file
  output_path = '../../data/detected_objects/' + str(video_idx).zfill(2) + '.pickle'
  with open(output_path, 'wb') as out_file:
    pickle.dump(detected_objects, out_file, protocol=pickle.HIGHEST_PROTOCOL)
  print('Output results to ' + output_path)
