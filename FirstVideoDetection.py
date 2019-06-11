from imageai.Detection import VideoObjectDetection
import os
import pickle

model_name = 'YOLOv3'

execution_path = os.getcwd()

detector = VideoObjectDetection()

if model_name is 'TinyYOLOv3':
  detector.setModelTypeAsTinyYOLOv3()
  detector.setModelPath(os.path.join(execution_path, 'yolo-tiny.h5'))

elif model_name is 'YOLOv3':
  detector.setModelTypeAsYOLOv3()
  detector.setModelPath(os.path.join(execution_path, 'yolo.h5'))

elif model_name is 'RetinaNet':
  detector.setModelTypeAsRetinaNet()
  detector.setModelPath(os.path.join(execution_path, 'resnet50_coco_best_v2.0.1.h5'))

else:
  raise ValueError('Unknown model ' + str(model_name))

detector.loadModel()

# # Generates a video displaying the detected objects
# output_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, 'traffic-mini.mp4'),
#                                              output_file_path=os.path.join(execution_path, 'traffic_mini_detected_' + model_name),
#                                              frames_per_second=29,
#                                              log_progress=True)

# Generates a record of all objects detected
all_frames = []
def forFrame(frame_number, output_array, output_count):
  all_frames.append(output_array)
  print('Frame #: ' , frame_number)
  # print('FOR FRAME ' , frame_number)
  # print('Output for each object : ', output_array)
  # print('Output count for unique objects : ', output_count)
  # print('------------END OF A FRAME --------------')
detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, 'traffic-mini.mp4'),
                                                    output_file_path=os.path.join(execution_path, 'video_frame_analysis_' + model_name),
                                                    frames_per_second=20,
                                                    per_frame_function=forFrame,
                                                    minimum_percentage_probability=30)

# Save results in pickle file
output_path = os.path.join(execution_path, 'video_frame_analysis_' + model_name + '.pickle')
with open(output_path, 'wb') as out_file:
  pickle.dump(all_frames, out_file, protocol=pickle.HIGHEST_PROTOCOL)
print('Output results to ' + output_path)
