"""This is the core object detector implementation."""

from imageai.Detection import VideoObjectDetection

def detect_objects(input_file_path, detection_confidence_threshold = 60):
  # detection_confidence_threshold (int between 1 and 99) is the minimum detector confidence needed to include an object
  
  # Create object detector based on RetinaNet and load model weights
  detector = VideoObjectDetection()
  detector.setModelTypeAsRetinaNet()
  detector.setModelPath('resnet50_coco_best_v2.0.1.h5')
  detector.loadModel()
  
  all_frames = []
  # Generates a record of all objects detected
  def forFrame(frame_number, output_array, output_count):
    all_frames.append(output_array)
    if frame_number % 100 == 0:
      print(input_file_path + ' Frame ' + str(frame_number))

  detector.detectObjectsFromVideo(input_file_path = input_file_path,
                                  output_file_path = 'labeled_video',
                                  frames_per_second = 30,
                                  per_frame_function = forFrame,
                                  minimum_percentage_probability = detection_confidence_threshold)
  return all_frames
