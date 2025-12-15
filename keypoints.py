import mediapipe as mp
import cv2 # opencv
import time
import numpy as np
import pandas as pd
import os

# CHANGE THIS TO UPDATE THE EXERCISE WE ARE EXTRACTING FROM
exercise = "RDL"

# key points + draw utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils # For drawing keypoints
points = mpPose.PoseLandmark._member_names_ # get names of landmarks
path = f"data/{exercise}" # dataset path, changes based on exercise

# create empty pandas dataframe
data = ["vid_num", "frame_num"]
for p in points:
        data.append(p + "_x")
        data.append(p + "_y")
        data.append(p + "_z")
        data.append(p + "_vis")
# data.append("vid_name")
data = pd.DataFrame(columns = data)

# extract landmarks from each video

cap = cv2.VideoCapture("videos/RDL_test3.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = "output_marked_test3.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
frame_count = 0
vid_df = pd.DataFrame(columns = data.columns)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    temp = [1, frame_count]

    # Process the frame with MediaPipe Pose
    result = pose.process(frame_rgb)

    # Draw the pose landmarks on the frame
    if result.pose_landmarks:
        mpDraw.draw_landmarks(frame, result.pose_landmarks, mpPose.POSE_CONNECTIONS)
        landmarks = result.pose_landmarks.landmark
        for i,j in zip(points,landmarks):
            temp = temp + [j.x, j.y, j.z, j.visibility]
        vid_df.loc[frame_count] = temp
        frame_count+=1

    # Display the frame (helpful for pictures on report)
    
    out.write(frame)
    cv2.imshow('MediaPipe Pose', frame)
    cv2.waitKey(2)

# bulk-insert one video into the dataframe
data = pd.concat([data, vid_df], ignore_index=True)

# save the data as a csv file, named after exercise
data.to_csv(f"keypoints/RDL_keypoints_test3.csv") 