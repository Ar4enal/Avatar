# Avatar
Android mp

# MediaPipe 
    input_stream: "input_video"
    output_stream: "output_video"
    output_stream: "POSE_LANDMARKS:pose_landmarks"
    output_stream: "POSE_ROI:pose_roi"
    output_stream: "LEFT_HAND_LANDMARKS:left_hand_landmarks"
    output_stream: "RIGHT_HAND_LANDMARKS:right_hand_landmarks"
    output_stream: "FACE_LANDMARKS:face_landmarks"
    output_stream: "iris_landmarks"

# Blendshape
use [eos via mediapipe](https://github.com/Ar4enal/eos/blob/master/python/test.py) to get facial blendshape.

# WebRTC
transmit audio to unity




