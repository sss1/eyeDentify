"""This module computes and reports summary statistics of the MOT videos."""
from collections import defaultdict
import pickle
import util

_VIDEOS = range(1, 15)
_DETECTION_DATA_DIR = '../../data/detected_objects'


def main():

  counts = defaultdict(int)
  distinct_classes = set()

  for video_idx in _VIDEOS:
    detection_data_fname = '{}/{}.pickle'.format(_DETECTION_DATA_DIR,
                                                 str(video_idx).zfill(2))
    print('Loading object detection data from {}...'
          .format(detection_data_fname))
    with open(detection_data_fname, 'rb') as in_file:
      all_frames = pickle.load(in_file)

    # Preprocess objects
    detected_video_objects = util.smooth_objects(all_frames)
    util.align_objects_to_screen(video_idx, detected_video_objects)

    for frame in detected_video_objects:
      counts['total_frames'] += 1
      distinct_classes_in_frame = set()
      for obj in frame:
        distinct_classes.add(obj.class_name)
        distinct_classes_in_frame.add(obj.class_name)
        try:
          counts[obj.class_name] += 1
        except KeyError:
          counts[obj.class_name] = 1
          counts['distinct_classes'] += 1
      for class_name in distinct_classes_in_frame:
        counts['frames_with_' + class_name] += 1

  print('{} distinct_classes: {}'
        .format(len(distinct_classes), distinct_classes))

  for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True):
    if 'frames_with_' in k:
      print('{}: {} ({}% of frames)'.format(k, v, 100*v/counts['total_frames']))

  for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True):
    if not 'frames_with_' in k:
      print('{}: {} ({}% of frames)'.format(k, v, 100*v/counts['total_frames']))


if __name__ == '__main__':
  main()
