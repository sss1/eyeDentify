"""Object tracking in a local video."""
from google.cloud import videointelligence
from google.protobuf.json_format import MessageToJson
import json

video_client = videointelligence.VideoIntelligenceServiceClient()
features = [videointelligence.enums.Feature.OBJECT_TRACKING]

path = '../../../data/MOT17_videos/01.mp4'

with open(path, 'rb') as file:
    input_content = file.read()

print('\nProcessing video for object annotations.')
operation = video_client.annotate_video(input_content=input_content, features=features)

result = operation.result(timeout=300)
print('\nFinished processing.\n')

# The first result is retrieved because a single video was processed.
object_annotations = MessageToJson(result)#.annotation_results[0].object_annotations)

with open('object_annotations.json', 'w') as f:
  f.write(object_annotations)

# # Get only the first annotation for demo purposes.
# object_annotation = object_annotations[0]
# print('Entity description: {}'.format(
#     object_annotation.entity.description))
# if object_annotation.entity.entity_id:
#     print('Entity id: {}'.format(object_annotation.entity.entity_id))
# 
# print('Segment: {}s to {}s'.format(
#     object_annotation.segment.start_time_offset.seconds +
#     object_annotation.segment.start_time_offset.nanos / 1e9,
#     object_annotation.segment.end_time_offset.seconds +
#     object_annotation.segment.end_time_offset.nanos / 1e9))
# 
# print('Confidence: {}'.format(object_annotation.confidence))
# 
# # Here we print only the bounding box of the first frame in this segment
# frame = object_annotation.frames[0]
# box = frame.normalized_bounding_box
# print('Time offset of the first frame: {}s'.format(
#     frame.time_offset.seconds + frame.time_offset.nanos / 1e9))
# print('Bounding box position:')
# print('\tleft  : {}'.format(box.left))
# print('\ttop   : {}'.format(box.top))
# print('\tright : {}'.format(box.right))
# print('\tbottom: {}'.format(box.bottom))
# print('\n')
