# -----------------------------------------------------------------------
#
# (c) Copyright 1997-2013, SensoMotoric Instruments GmbH
# 
# Permission  is  hereby granted,  free  of  charge,  to any  person  or
# organization  obtaining  a  copy  of  the  software  and  accompanying
# documentation  covered  by  this  license  (the  "Software")  to  use,
# reproduce,  display, distribute, execute,  and transmit  the Software,
# and  to  prepare derivative  works  of  the  Software, and  to  permit
# third-parties to whom the Software  is furnished to do so, all subject
# to the following:
# 
# The  copyright notices  in  the Software  and  this entire  statement,
# including the above license  grant, this restriction and the following
# disclaimer, must be  included in all copies of  the Software, in whole
# or  in part, and  all derivative  works of  the Software,  unless such
# copies   or   derivative   works   are   solely   in   the   form   of
# machine-executable  object   code  generated  by   a  source  language
# processor.
# 
# THE  SOFTWARE IS  PROVIDED  "AS  IS", WITHOUT  WARRANTY  OF ANY  KIND,
# EXPRESS OR  IMPLIED, INCLUDING  BUT NOT LIMITED  TO THE  WARRANTIES OF
# MERCHANTABILITY,   FITNESS  FOR  A   PARTICULAR  PURPOSE,   TITLE  AND
# NON-INFRINGEMENT. IN  NO EVENT SHALL  THE COPYRIGHT HOLDERS  OR ANYONE
# DISTRIBUTING  THE  SOFTWARE  BE   LIABLE  FOR  ANY  DAMAGES  OR  OTHER
# LIABILITY, WHETHER  IN CONTRACT, TORT OR OTHERWISE,  ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE  SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# -----------------------------------------------------------------------

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
from datetime import datetime
from iViewXAPI import  *            #iViewX library
from iViewXAPIReturnCodes import *
import threading
import time
import cv2 # Use OpenCV to display video
import screeninfo
import pickle

# ---------------------------------------------
#---- connect to iViewX
# ---------------------------------------------

# Ask user for participant number/id
participant_id = raw_input('Enter participant ID: ')
today = `datetime.now().month` + "_" + `datetime.now().day` + "_" + `datetime.now().year`
time_now = `datetime.now().hour` + "_" + `datetime.now().minute`
eyetracking_file_name = participant_id + "_eyetracking_" + today + "_" + time_now + '.csv'
print("Eye-tracking data will save to the file: " + eyetracking_file_name)
eyetracking_writer = csv.writer(open(eyetracking_file_name,'wb'), delimiter = ',')
header = ["ComputerClock_Timestamp", "Avg_GazeX", "Avg_GazeY", "LeftEye_GazeX", "LeftEye_GazeY", "RightEye_GazeX", "RightEye_GazeY", "LeftEye_Diam", "RightEye_Diam"]
eyetracking_writer.writerow(header)

print("Preparing eye-tracker...")

res = iViewXAPI.iV_SetLogger(c_int(1), c_char_p("iViewXSDK_Python_SimpleExperiment.txt"))
res = iViewXAPI.iV_Connect(c_char_p('127.0.0.1'), c_int(4444), c_char_p('127.0.0.1'), c_int(5555))
if res != 1:
    HandleError(res)
    exit(0)

res = iViewXAPI.iV_GetSystemInfo(byref(systemData))

# ---------------------------------------------
#---- define callback function that records eye-tracking data and displays video
# ---------------------------------------------

# Realign eye-tracking with origin of video.
def realign(x, y):
  return (x - xPos, y - yPos)

def SampleCallback(sample):

  # Eye-tracking timestamp (relative to video onset) in milliseconds
  timestamp = time.time()

  # Record eye-tracking data
  leftX = sample.leftEye.gazeX
  leftY = sample.leftEye.gazeY
  rightX = sample.rightEye.gazeX
  rightY = sample.rightEye.gazeY
  leftDiam = sample.leftEye.diam
  rightDiam = sample.rightEye.diam

  # Recode missing data from 0.0 to NaN.
  # Take average of left and right eyes, unless one or both eyes is missing.
  if leftX > 0.0 and rightX > 0.0: # Both eyes captured
    leftX, leftY = realign(leftX, leftY)
    rightX, rightY = realign(rightX, rightY)
    bestX = (leftX + rightX) / 2.0
    bestY = (leftY + rightY) / 2.0

  elif leftX > 0.0: # Right eye is missing; use left
    leftX, leftY = realign(leftX, leftY)
    rightX = rightY = float('nan')
    bestX = leftX
    bestY = leftY

  elif rightX > 0.0: # Left eye is missing; use right
    leftX = leftY = float('nan')
    rightX, rightY = realign(rightX, rightY)
    bestX = rightX
    bestY = rightY

  else: # Both eyes missing; report NaN
    bestX = bestY = leftX = leftY = rightX = rightY = float('nan')

  # Sometimes, gaze data is missing but pupil diameter is not (or vice versa);
  # hence, recode missing pupil diameter data as NaN separately
  if leftDiam == 0.0:
    leftDiam = float('nan')
  if rightDiam == 0.0:
    rightDiam = float('nan')

  # Write timepoint data as new row in output file
  eyetracking_writer.writerow([str(x) for x in [timestamp, bestX, bestY, leftX, leftY, rightX, rightY, leftDiam, rightDiam]])

  return 0

def EventCallback(event):
  return 0

# ---------------------------------------------
#---- implement background thread to run above CallBack
# ---------------------------------------------

CMPFUNC = WINFUNCTYPE(c_int, CSample)
smp_func = CMPFUNC(SampleCallback)
sampleCB = False

CMPFUNC = WINFUNCTYPE(c_int, CEvent)
event_func = CMPFUNC(EventCallback)
eventCB = False

class StoppableThread(threading.Thread):

  def __init__(self):
    threading.Thread.__init__(self)

  def run(self):
    self.run = True
    while self.run:
      res = iViewXAPI.iV_SetSampleCallback(smp_func)
      sampleCB = True
      res = iViewXAPI.iV_SetEventCallback(event_func)
      eventCB = True

  def stop(self):
    self.run = False

# ---------------------------------------------
#---- UI to start/stop DataStreaming
# ---------------------------------------------

command = None
while (command != 'start'):
  command = raw_input("Type 'start' and press Enter to begin experiment: ")





stimulus_record_file_name = participant_id + "_stimulus_" + today + "_" + time_now + '.csv'
videos_list = np.random.permutation(range(14)) + 1
print("Stimulus data data will save to the file: " + stimulus_record_file_name)

with open(stimulus_record_file_name, 'w', newline='') as csvfile:
  stimulus_writer = csv.writer(data, delimiter = ',')
  header = ["ComputerClock_Timestamp", "Video", "Frame", "Target_Name", "startX", "startY", "endX", "endY"]
  stimulus_writer.writerow(header)
  for video_ID in video_list:
    video_file_name = 'data/MOT17_videos/' + str(video_ID).zfill(2) + '.mp4'
    object_file_name = 'data/detected_objects/' + str(video_ID).zfill(2) + '.pickle'

    # ---------------------------------------------
    # ---- set up video display
    # ---------------------------------------------

    video = cv2.VideoCapture(video_file_name)
    with open(object_file_name, 'rb') as in_file:
      all_frames = pickle.load(in_file)
    
    # Get basic video information
    video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    FPS = video.get(cv2.CAP_PROP_FPS) # natural frame rate
    
    # Set the inter-frame delay based on the video's natural framerate
    delay = 1.0/FPS
    
    # Create video display window and center it on the screen
    screen = screeninfo.get_monitors()[0]
    # (xPos, yPos) is the upper-left corner of the video display
    xPos = int((screen.width - video_width) / 2.0)
    yPos = int((screen.height - video_height) / 2.0)
    if xPos < 0 or yPos < 0:
      print('Warning: video size ' + str((video_width, video_height)) + \
            ' is larger than screen size ' + str((screen.width, screen.height)) + \
            '.')
      print('Some of the video may be cut off.')
    cv2.namedWindow('Video Frame')
    cv2.moveWindow('Video Frame', xPos, yPos)
    
    # Set up to display the first frame
    current_frame = 0
    videoStartTime = time.time()
    nextFrameExists, frame = video.read() # Load first video frame
    
    print("Initiating eye-tracking...")
    # start recording in a background thread
    thr = StoppableThread()
    thr.start()

    print("Playing video...")
    while nextFrameExists: # While there are more frames to display, continue displaying video
      if time.time() > videoStartTime + current_frame * delay:
          cv2.imshow('Video Frame', frame) # Display frame
          cv2.waitKey(1)
          nextFrameExists, frame = video.read() # Load next video frame
          target = target_list[current_frame]
          b = target['box_points']
          cv2.ellipse(frame, centroid, ((b[2] - b[0])//2, (b[3] - b[1])//2), 0, 0, 360, color = (0, 255, 0), thickness = 2)
          stimulus_writer.writerow([str(x) for x in [time.time(), video_ID, current_frame, target['name'], b[0], b[1], b[2], b[3]]])
          current_frame += 1

    video.release()
    cv2.destroyAllWindows()

    print("Terminating eye-tracking collection...")
    # tell background thread to stop and wait for it to terminate
    thr.stop()
    thr.join()


# wait for user to terminate recording
# command = 0
# while (command != 'quit'):
#   command = raw_input("Type 'quit' and press Enter to terminate eye-tracking recording: ")

# ---------------------------------------------
#---- stop recording and disconnect from iViewX
# ---------------------------------------------

print("Disconnecting from eye-tracker...")

res = iViewXAPI.iV_Disconnect()

print("Saving eye-tracking data...")

data.flush()
data.close()

print("Done!")
