# eyeDentify

## Project Overview

This repository contains code and data for a project on Semantic Eye-Tracking
(SET), that is, using eye-tracking to identify the object a person is visually
attending to in a dynamic visual scene.

A SET algorithm takes two main inputs:

To evaluate SET algorithms, we need three datasets:
1) Stimulus video: a first-person video recording of the participant's
   environment.
2) Eye-tracking: this consists of a sequence of (t, x, y) points, denoting the
   (x, y)-coordinates of the participants gaze at time t.

In this project, we assume that the datasets for 1) and 2) are aligned
temporally and spatially, although, in general, this alignment could be
performed in an additional step. Given these inputs, the SET algorithm outputs
*predicted labels*, i.e., predictions, in each video frame, of the object to
which the participant is attending. In this project, we assume that the
participant's attention is *overt*, i.e., that their gaze is directed at the
object to which they are attending.

To evaluate a SET algorithm, we additionally need:

3) Ground truth labels, which indicate the *true* object a participant is
attending to at each point in time.

While 1) and 2) are widely available, 3) is challenging to obtain. In practice,
3) is typically obtained by having trained human coders watch the stimulus
videos, overlaid with eye-tracking, and manually label the objects of attention
throughout the video. This is quite time-consuming and subjective, reducing
replicability of experimental results. Hence, we would like to automate this
process.

So far, we have conducted a "Guided Viewing" experiment, in which we collected
data from adult participants watching videos while we collected eye-tracking
data, in a controlled environment. In each video frame, a single Target object
(randomly in real time for each participant) was identified by a bright green
bounding box, and we asked participants to continuously, overtly track this
labeled Target object. To ensure that we observed attentional transitions, the
Target object changed at random intervals throughout the video. In this
experiment, we assume that the particpant is attending to the labeled Target
object, at each point in time (excluding a short transition period whenever the
Target changes).

For this study, we used 14 publicly available videos from the
[MOT17 Challenge](https://motchallenge.net/data/MOT17/).

Eyetracking data and ground truth labels are not currently included in this
repository, due to GitHub file size limits. This will be uploaded elsewhere.

## Technical Notes
This project is compatible with Python 3.9.5. Necessary packages are listed in
`requirements.txt` and can be installed using
```
pip install -r requirements.txt
```

## File overview
The main codebase is in the `code/` directory. The most important code files are
probably:

`experiment1.py` runs an evaluattion of the HMM algorithm.

`visualizer.py` generates and plays a video of the stimulus, overlaid with
bounding boxes of the detected objects and the participants gaze.

`load_and_preprocess_data.py` contains a `load_participant()` function that
loads the eye-tracking and stimulus data for each participant.

`display_experiment.py` was the original experiment script used to display the
stimulus.

`hmm.py` contains the main code of the HMM algorithm.
