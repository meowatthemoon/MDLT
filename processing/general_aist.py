import pickle
import os

from config import ANNOTATION_DIR, SMPL_DIR

import numpy as np
import torch

from aist_plusplus.loader import AISTDataset
from smplx import SMPL

# ------ Start of Keypoint Functions
def __get_rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s < 1e-12:
        # Handle the case where s is very small to avoid division by near-zero
        rotation_matrix = np.eye(3)
    else:
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    return rotation_matrix

def __align_spine_and_shoulders(keypoints : np.array) -> np.array:
    # Calculate Center
    right_shoulder = keypoints[2]
    left_shoulder = keypoints[3]
    center = (right_shoulder + left_shoulder) / 2

    # Subtract all keypoints by center
    for keypoint in keypoints:
        keypoint -= center

    # Align Shoulders
    A = left_shoulder = keypoints[3]
    B = desired_unit = np.array([1, 0, 0])
    mat = __get_rotation_matrix_from_vectors(A, B) * -1
    keypoints = np.array([mat.dot(keypoint) for keypoint in keypoints])

    # Align Spine
    A = pelvis = keypoints[0]
    B = desired_unit = np.array([0, -1, 0])
    mat = __get_rotation_matrix_from_vectors(A, B)
    keypoints = np.array([mat.dot(keypoint) for keypoint in keypoints])

    # Re-add center
    keypoints = np.array([keypoint + center for keypoint in keypoints])

    return keypoints

def __to_unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def __angle_between_vectors(vector_1 : np.array, vector_2 : np.array) -> float:
    # Calculate dot product of v1 and v2
    dot_product = np.dot(vector_1, vector_2)

    # Calculate norms of v1 and v2
    norm_u = np.linalg.norm(vector_1)
    norm_v = np.linalg.norm(vector_2)

    # Calculate cosine of the angle
    cos_theta = dot_product / (norm_u * norm_v)

    # Calculate the angle in radians
    theta_rad = np.arccos(cos_theta)

    return theta_rad

def __get_joint_angles_from_arm_keypoints(arm_keypoints : np.array) -> np.array:
    arm_vector = np.array([arm_keypoints[1][0] - arm_keypoints[0][0], arm_keypoints[1][1] - arm_keypoints[0][1], arm_keypoints[1][2] - arm_keypoints[0][2]])
    arm_vector = __to_unit_vector(arm_vector)

    forearm_vector = np.array([arm_keypoints[2][0] - arm_keypoints[1][0], arm_keypoints[2][1] - arm_keypoints[1][1], arm_keypoints[2][2] - arm_keypoints[1][2]])
    forearm_vector = __to_unit_vector(forearm_vector)

    hand_vector = np.array([arm_keypoints[3][0] - arm_keypoints[2][0], arm_keypoints[3][1] - arm_keypoints[2][1], arm_keypoints[3][2] - arm_keypoints[2][2]])
    hand_vector = __to_unit_vector(hand_vector)

    # Angle 1
    arm_y_z = np.array([arm_vector[1], arm_vector[2]])
    angle_1 = __angle_between_vectors(vector_1 = arm_y_z, vector_2 = np.array([-1, 0]))

    # Angle 2
    arm_x_y = np.array([arm_vector[0], arm_vector[1]])
    angle_2 = __angle_between_vectors(vector_1 = arm_y_z, vector_2 = np.array([0, -1]))
    angle_2 = angle_2 * -1

    # Angle 3
    angle_3 = __angle_between_vectors(vector_1 = arm_vector, vector_2 = forearm_vector)
    angle_3 = angle_3 * -1

    # Angle 4
    angle_4 = __angle_between_vectors(vector_1 = forearm_vector, vector_2 = hand_vector)
    angle_4 = angle_4 * -1
    
    return np.array([angle_1, angle_2, angle_3, angle_4])

def __get_angles_from_keypoints(keypoints : np.array):
    keypoints = __align_spine_and_shoulders(keypoints = keypoints)

    # Right Arm Only
    right_arm_keypoints = np.array([keypoints[3], keypoints[5], keypoints[7], keypoints[9]])

    angles = __get_joint_angles_from_arm_keypoints(arm_keypoints = right_arm_keypoints)
    return angles

def __get_joint_angles_from_kp_sequence(keypoints_sequence : np.array) -> np.array:
    kp_idx = [0, 3, 16, 17, 18, 19, 20, 21, 35, 42]
    keypoints_sequence = keypoints_sequence[:, kp_idx, :]

    angles = np.array([__get_angles_from_keypoints(keypoints = keypoints) for keypoints in keypoints_sequence])
    return angles


def __load_keypoints_from_sequence(sequence_name : str) -> np.array:
    aist_dataset = AISTDataset(ANNOTATION_DIR)

    smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(aist_dataset.motion_dir, sequence_name)
    smpl = SMPL(model_path = SMPL_DIR, gender = 'MALE', batch_size = 1)
    keypoints3d = smpl.forward(
        global_orient = torch.from_numpy(smpl_poses[:, 0:1]).float(),
        body_pose = torch.from_numpy(smpl_poses[:, 1:]).float(),
        transl = torch.from_numpy(smpl_trans).float(),
        scaling = torch.from_numpy(smpl_scaling.reshape(1, 1)).float(),
    ).joints.detach().numpy()

    return keypoints3d

def get_arm_joint_angles_from_sequence(sequence_name : str) -> np.array:
    keypoints_sequence = __load_keypoints_from_sequence(sequence_name = sequence_name)
    angles = __get_joint_angles_from_kp_sequence(keypoints_sequence = keypoints_sequence)
    
    return angles

# ------ End of Keypoint Functions

# ------ Start of Timetamp Function

def load_timestamps(sequence_name : str) -> np.array:
    with open(os.path.join(ANNOTATION_DIR, "keypoints2d", f"{sequence_name}.pkl"), 'rb') as pickle_file:
        content = pickle.load(pickle_file)

        timestamps_micro = content["timestamps"]

    return np.array(timestamps_micro)

# ------ End of Timestamp Function

# ------ Start of Synchronization Function

def synchronize(audio_features : np.array, joint_angles : np.array, timestamps_micro : np.array, fps : int) -> tuple:
    X, Y, T = [], [], []

    N = len(timestamps_micro)

    t = 0
    for i, audio_feature in enumerate(audio_features):
        audio_micro_seconds = (i + 1) * 1 / fps * 1000 * 1000

        while True:
            # No more timestamps left
            if t == N -1: 
                break

            # If I'm closer to the current one than to the next one
            if abs(audio_micro_seconds - timestamps_micro[t]) < abs(audio_micro_seconds - timestamps_micro[t + 1]):
                break

            t += 1

        X.append(audio_feature)
        Y.append(joint_angles[t])
        T.append(timestamps_micro[t])

    return np.array(X), np.array(Y), np.array(T)

# ------ End of Synchronization Function
