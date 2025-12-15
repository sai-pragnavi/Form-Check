import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# from dtw import *
from dtaidistance import dtw
from scipy.stats import pearsonr
import cv2
import csv

def calc_angle(v1, v2):
    den = (np.linalg.norm(v1.T) * np.linalg.norm(v2.T))
    dot = np.sum(v1.T * v2.T, axis = 1)
    angle_radians = np.arccos(dot / den)
    angle_deg = np.degrees(angle_radians)
    return angle_deg

def calc_vec(v1, v2, side):
    vec = np.array([v2[side][0]-v1[side][0],
                    v2[side][1]-v1[side][1]
                    ])
    return vec

def lower_body_score():
    data_train = pd.read_csv(f"keypoints/RDL_keypoints_train.csv")
    data_test = pd.read_csv(f"keypoints/RDL_keypoints_test2.csv")
    vid_01 = np.array(data_train)
    vid_02 = np.array(data_test)
    nose_keypoints_01 = [vid_01[:,3], vid_01[:,4]]
    nose_keypoints_02 = [vid_02[:,3], vid_02[:,4]]
        
    shoulder_keypoints_01 = { 
        "Left": [vid_01[:,47], vid_01[:,48], vid_01[:,49], vid_01[:,50]],
        "Right": [vid_01[:,51], vid_01[:,52], vid_01[:,53], vid_01[:,54]] 
        } 
    shoulder_keypoints_02 = { 
        "Left": [vid_02[:,47], vid_02[:,48], vid_02[:,49], vid_02[:,50]],
        "Right": [vid_02[:,51], vid_02[:,52], vid_02[:,53], vid_02[:,54]] 
        } 
    elbow_keypoints_01 = { 
        "Left": [vid_01[:,54], vid_01[:,55], vid_01[:,56], vid_01[:,57]],
        "Right": [vid_01[:,58], vid_01[:,59], vid_01[:,60], vid_01[:,61]] 
        } 
    elbow_keypoints_02 = { 
        "Left": [vid_02[:,54], vid_02[:,55], vid_02[:,56], vid_02[:,57]],
        "Right": [vid_02[:,58], vid_02[:,59], vid_02[:,60], vid_02[:,61]]
        } 
    wrist_keypoints_01 = { 
        "Left": [vid_01[:,62], vid_01[:,63], vid_01[:,64], vid_01[:,65]],
        "Right": [vid_01[:,66], vid_01[:,67], vid_01[:,68], vid_01[:,69]] 
        } 
    wrist_keypoints_02 = { 
        "Left": [vid_02[:,62], vid_02[:,63], vid_02[:,64], vid_02[:,65]],
        "Right": [vid_02[:,66], vid_02[:,67], vid_02[:,68], vid_02[:,69]] 
        } 
    hip_keypoints_01 = { 
        "Left": [vid_01[:,95], vid_01[:,96], vid_01[:,97], vid_01[:,98]],
        "Right": [vid_01[:,99], vid_01[:,100], vid_01[:,101], vid_01[:,102]] 
        } 
    hip_keypoints_02 = { 
        "Left": [vid_02[:,95], vid_02[:,96], vid_02[:,97], vid_02[:,98]],
        "Right": [vid_02[:,99], vid_02[:,100], vid_02[:,101], vid_02[:,102]] 
        } 
    knee_keypoints_01 = { 
        "Left": [vid_01[:,103], vid_01[:,104], vid_01[:,105], vid_01[:,106]],
        "Right": [vid_01[:,107], vid_01[:,108], vid_01[:,109], vid_01[:,110]] 
        } 
    knee_keypoints_02 = { 
        "Left": [vid_02[:,103], vid_02[:,104], vid_02[:,105], vid_02[:,106]],
        "Right": [vid_02[:,107], vid_02[:,108], vid_02[:,109], vid_02[:,110]]  
        } 
    ankle_keypoints_01 = { 
        "Left": [vid_01[:,111], vid_01[:,112], vid_01[:,113], vid_01[:,114]],
        "Right": [vid_01[:,115], vid_01[:,116], vid_01[:,117], vid_01[:,118]] 
        } 
    ankle_keypoints_02 = { 
        "Left": [vid_02[:,111], vid_02[:,112], vid_02[:,113], vid_02[:,114]],
        "Right": [vid_02[:,115], vid_02[:,116], vid_02[:,117], vid_02[:,118]] 
        } 
    

    #angle between hip -> knee and knee -> ankle
    hip_knee_vec_01_left = calc_vec(hip_keypoints_01, knee_keypoints_01, "Left")
    hip_knee_vec_01_right = calc_vec(hip_keypoints_01, knee_keypoints_01, "Right")
    hip_knee_vec_02_left = calc_vec(hip_keypoints_02, knee_keypoints_02, "Left")
    hip_knee_vec_02_right = calc_vec(hip_keypoints_02, knee_keypoints_02, "Right")
    
    knee_ankle_vec_01_left = calc_vec(knee_keypoints_01, ankle_keypoints_01, "Left")
    knee_ankle_vec_01_right = calc_vec(knee_keypoints_01, ankle_keypoints_01, "Right")
    knee_ankle_vec_02_left = calc_vec(knee_keypoints_02, ankle_keypoints_02, "Left")
    knee_ankle_vec_02_right = calc_vec(knee_keypoints_02, ankle_keypoints_02, "Right")
    
    hip_knee_ankle_left_train = calc_angle(hip_knee_vec_01_left, knee_ankle_vec_01_left)
    print("Angle between hip, knee and knee, ankle left for train is ",hip_knee_ankle_left_train)
    hip_knee_ankle_right_train = calc_angle(hip_knee_vec_01_right, knee_ankle_vec_01_right)
    print("Angle between hip, knee and knee, ankle right for train is ",hip_knee_ankle_right_train)
    
    hip_knee_ankle_left_test = calc_angle(hip_knee_vec_02_left, knee_ankle_vec_02_left)
    print("Angle between hip, knee and knee, ankle left for test is ",hip_knee_ankle_left_test)
    hip_knee_ankle_right_test = calc_angle(hip_knee_vec_02_right, knee_ankle_vec_02_right)
    print("Angle between hip, knee and knee, ankle right for test is ",hip_knee_ankle_right_test)
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(hip_knee_ankle_right_train, label ='Right train', color='red')
    ax.plot(hip_knee_ankle_left_train, label='left train', color='black')
    ax.plot(hip_knee_ankle_right_test, label ='Right test', color='orange')
    ax.plot(hip_knee_ankle_left_test, label='Left test', color='blue')
    ax.legend()
    ax.set_title("Hip, Knee, Ankle")
    plt.show()

    hka_distance_right = dtw.distance(hip_knee_ankle_right_train, hip_knee_ankle_right_test, use_c=False)
    print(hka_distance_right)
    hka_distance_left = dtw.distance(hip_knee_ankle_left_train, hip_knee_ankle_left_test, use_c=False)
    print(hka_distance_left)

    #angle between shoulder -> back and back -> hip
    sh_center_train = [0.5*(shoulder_keypoints_01['Left'][0]+shoulder_keypoints_01['Right'][0]), 
                 0.5*(shoulder_keypoints_01['Left'][1]+shoulder_keypoints_01['Right'][1])]
    hip_center_train = [0.5*(hip_keypoints_01['Left'][0]+hip_keypoints_01['Right'][0]),
                         0.5*(hip_keypoints_01['Left'][1]+hip_keypoints_01['Right'][1])]
    torso_center_train = [0.5*(sh_center_train[0]+hip_center_train[0]),
                     0.5*(sh_center_train[1]+hip_center_train[1])] 
    v1_train = np.array(hip_center_train) - np.array(torso_center_train)
    v2_train = np.array(torso_center_train) - np.array(sh_center_train)

    sh_center_test = [0.5*(shoulder_keypoints_02['Left'][0]+shoulder_keypoints_02['Right'][0]), 
                 0.5*(shoulder_keypoints_02['Left'][1]+shoulder_keypoints_02['Right'][1])]
    hip_center_test = [0.5*(hip_keypoints_02['Left'][0]+hip_keypoints_02['Right'][0]),
                         0.5*(hip_keypoints_02['Left'][1]+hip_keypoints_02['Right'][1])]
    torso_center_test = [0.5*(sh_center_test[0]+hip_center_test[0]), 
                         0.5*(sh_center_test[1]+hip_center_test[1])] 
    v1_test = np.array(hip_center_test) - np.array(torso_center_test)
    v2_test = np.array(torso_center_test) - np.array(sh_center_test)
    sh_back_hip_train = calc_angle(v1_train, v2_train)
    print("Angle between Hip, Torso, Nose for train is ",sh_back_hip_train)
    sh_back_hip_test = calc_angle(v1_test, v2_test)
    print("Angle between  Hip, Torso, Nose for test is ",sh_back_hip_test)
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(sh_back_hip_train, label ='Train', color='red')
    ax.plot(sh_back_hip_test, label='Test', color='black')
    ax.legend()
    ax.set_title(" Hip, Torso, Nose")
    plt.show()

    sbh_distance_right = dtw.distance(sh_back_hip_train, sh_back_hip_test, use_c=False)
    print(sbh_distance_right)
    


    #angle between shoulder -> elbow and elbow -> wrist
    shoulder_elbow_vec_01_left = calc_vec(shoulder_keypoints_01, elbow_keypoints_01, "Left")
    shoulder_elbow_vec_01_right = calc_vec(shoulder_keypoints_01, elbow_keypoints_01, "Right")
    shoulder_elbow_vec_02_left = calc_vec(shoulder_keypoints_02, elbow_keypoints_02, "Left")
    shoulder_elbow_vec_02_right = calc_vec(shoulder_keypoints_02, elbow_keypoints_02, "Right")

    elbow_wrist_vec_01_left = calc_vec(elbow_keypoints_01,wrist_keypoints_01, "Left")
    elbow_wrist_vec_01_right = calc_vec(elbow_keypoints_01, wrist_keypoints_01, "Right")
    elbow_wrist_vec_02_left = calc_vec(elbow_keypoints_02, wrist_keypoints_02, "Left")
    elbow_wrist_vec_02_right = calc_vec(elbow_keypoints_02, wrist_keypoints_02, "Right")
    
    sh_elbow_wrist_left_train = calc_angle(shoulder_elbow_vec_01_left, elbow_wrist_vec_01_left)
    print("Angle between Shoulder, Elbow, Wrist left for train is ",sh_elbow_wrist_left_train)
    sh_elbow_wrist_right_train = calc_angle(shoulder_elbow_vec_01_right, elbow_wrist_vec_01_right)
    print("Angle between Shoulder, Elbow, Wrist right for train is ",sh_elbow_wrist_right_train)
    
    sh_elbow_wrist_left_test = calc_angle(shoulder_elbow_vec_02_left, elbow_wrist_vec_02_left)
    print("Angle between Shoulder, Elbow, Wrist left for test is ",sh_elbow_wrist_left_test)
    sh_elbow_wrist_right_test = calc_angle(shoulder_elbow_vec_02_right, elbow_wrist_vec_02_right)
    print("Angle between Shoulder, Elbow, Wrist right for test is ",sh_elbow_wrist_right_test)
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(sh_elbow_wrist_right_train, label ='Right train', color='red')
    ax.plot(sh_elbow_wrist_left_train, label='left train', color='black')
    ax.plot(sh_elbow_wrist_right_test, label ='Right test', color='orange')
    ax.plot(sh_elbow_wrist_left_test, label='Left test', color='blue')
    ax.legend()
    ax.set_title("Shoulder, Elbow, Wrist")
    plt.show()

    sew_distance_right = dtw.distance(sh_elbow_wrist_right_train, sh_elbow_wrist_right_test, use_c=False)
    print(sew_distance_right)
    sew_distance_left = dtw.distance(sh_elbow_wrist_left_train, sh_elbow_wrist_left_test, use_c=False)
    print(sew_distance_left)
    
if __name__ == "__main__":
    print("Calculate workout efficiency for RDL")
    lower_body_score()