import pickle

def calc_centroid(obj):
  # Recall that, in Python 3, division is float by default
  return ((obj['box_points'][0] + obj['box_points'][2])/2,
          (obj['box_points'][1] + obj['box_points'][3])/2)

def smooth_objects(all_frames, spatial_tolerance, temporal_tolerance):
  # Since we assume that objects cannot change types, we separately run the
  # object tracking algorithm for each object type
  unique_obj_types = set([obj['name'] for frame in all_frames for obj in frame])
  print(unique_obj_types)

  for obj_type in unique_obj_types:
    for frame in all_frames:
      print(obj_type + ': ' + str([calc_centroid(obj) for obj in frame if obj['name'] is obj_type]))
    # all_frames_centroids_by_type = [[calc_centroid(obj) for obj in frame if obj['name'] is obj_type] for frame in all_frames]

with open('video_frame_analysis_YOLOv3.pickle', 'rb') as in_file:
  all_frames = pickle.load(in_file)

smooth_objects(all_frames,2,3)
