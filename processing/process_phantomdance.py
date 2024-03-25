import json
import math
import os

import essentia
#import essentia.streaming
from essentia.standard import *
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from extractor import FeatureExtractor

FPS = 30
HOP_LENGTH = 512
SR = FPS * HOP_LENGTH

DATASET_PATH = '<path_to_phantom>/PhantomDanceDatav1.1'
OUTPUT_DIR = f"./phantom_data_438_{FPS}fps"

def __get_audio_features(audio_path : str) -> np.array:
    loader = None
    try:
        loader = essentia.standard.MonoLoader(filename = audio_path, sampleRate = SR)
    except RuntimeError:
        return None

    audio = loader()
    audio = np.array(audio).T

    extractor = FeatureExtractor()
    melspe_db = extractor.get_melspectrogram(audio, SR)
    
    mfcc = extractor.get_mfcc(melspe_db)
    mfcc_delta = extractor.get_mfcc_delta(mfcc)
    # mfcc_delta2 = get_mfcc_delta2(mfcc)

    audio_harmonic, audio_percussive = extractor.get_hpss(audio)
    # harmonic_melspe_db = get_harmonic_melspe_db(audio_harmonic, sr)
    # percussive_melspe_db = get_perc ussive_melspe_db(audio_percussive, sr)
    chroma_cqt = extractor.get_chroma_cqt(audio_harmonic, SR, octave= 7)#7 if SR ==15360*2 else 5)
    # chroma_stft = extractor.get_chroma_stft(audio_harmonic, sr)

    onset_env = extractor.get_onset_strength(audio_percussive, SR)
    tempogram = extractor.get_tempogram(onset_env, SR)
    onset_beat = extractor.get_onset_beat(onset_env, SR)[0]
    # onset_tempo, onset_beat = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    # onset_beats.append(onset_beat)

    onset_env = onset_env.reshape(1, -1)

    feature = np.concatenate([
        # melspe_db,
        mfcc, # 20
        mfcc_delta, # 20
        # mfcc_delta2,
        # harmonic_melspe_db,
        # percussive_melspe_db,
        # chroma_stft,
        chroma_cqt, # 12
        onset_env, # 1
        onset_beat, # 1
        tempogram
    ], axis=0)

    feature = feature.transpose(1, 0)
    #print(f'acoustic feature -> {feature.shape}')

    return feature

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
    left_shoulder = keypoints[1]
    center = (right_shoulder + left_shoulder) / 2

    # Subtract all keypoints by center
    for keypoint in keypoints:
        keypoint -= center

    # Align Shoulders
    A = left_shoulder = keypoints[1]
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

def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix.

    Parameters:
    q (numpy.ndarray): Quaternion array.

    Returns:
    numpy.ndarray: Rotation matrix.
    """
    w, x, y, z = q
    rotation_matrix = np.array([[1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
                                [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
                                [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]])
    return rotation_matrix

def smpl_to_3d_keypoints(smpl_json_file):
    with open(smpl_json_file, 'r') as f:
        smpl_data = json.load(f)

    #bone_names = smpl_data['bone_name']
        
    desired_bone_names = [
        "Pelvis",
        "L_Shoulder",
        "R_Shoulder",
        "R_Elbow",
        "R_Wrist",
        "R_Hand"
    ]

    root_positions = np.array(smpl_data['root_positions'])
    rotations = np.array(smpl_data['rotations'])
    bone_names = np.array(smpl_data['bone_name'])

    assert len(rotations) == len(root_positions), "Root positions and Rotations have different lengths"

    num_frames = len(root_positions)
    num_joints = len(bone_names)

    keypoints_3d = []

    for i in range(num_frames):
        frame_keypoints = []

        for j in range(num_joints):
            bone_name = bone_names[j]
            if bone_name not in desired_bone_names:
                continue

            rotation_matrix = quaternion_to_rotation_matrix(rotations[i][j])

            # Calculate endpoint of the bone
            end_joint_position = root_positions[i] + np.dot(rotation_matrix, np.array([0, 0, j + 1]))

            """
            rotation_quaternion = rotations[i][j]

            # Convert quaternion to rotation matrix
            rotation_matrix = Rotation.from_quat(rotation_quaternion).as_matrix()

            # Get bone length (assuming it's constant for each bone, or you need to provide bone lengths)
            # Here, you can use a lookup table or hardcoded values
            bone_length = 1.0

            # Compute the position of the end joint
            end_joint_position = root_positions[i] + np.dot(rotation_matrix, [0, 0, bone_length])
            """            

            frame_keypoints.append(end_joint_position)

        keypoints_3d.append(frame_keypoints)

    assert not np.isnan(keypoints_3d).any(), "nan in smpl conversion"
    assert not np.isinf(keypoints_3d).any(), "inf in smpl conversion"

    return np.array(keypoints_3d)

def __to_unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

"""def normalize_angle(angle):
    normalized_angle = np.mod(angle + np.pi, 2*np.pi) - np.pi
    return normalized_angle"""


def __angle_between_vectors(vector_1 : np.array, vector_2 : np.array) -> float:
    # Calculate dot product of v1 and v2
    dot_product = np.dot(vector_1, vector_2)
    if round(dot_product, 2) == 1.00:
        return 0

    # Calculate norms of v1 and v2
    norm_u = np.linalg.norm(vector_1)
    norm_v = np.linalg.norm(vector_2)

    # Calculate cosine of the angle
    cos_theta = dot_product / (norm_u * norm_v)

    # Calculate the angle in radians
    theta_rad = np.arccos(cos_theta)

    # Normalize
    #theta_rad = normalize_angle(theta_rad)

    return theta_rad

def __get_joint_angles_from_arm_keypoints(arm_keypoints : np.array) -> np.array:
    arm_vector = np.array([arm_keypoints[1][0] - arm_keypoints[0][0], arm_keypoints[1][1] - arm_keypoints[0][1], arm_keypoints[1][2] - arm_keypoints[0][2]])
    arm_vector = __to_unit_vector(arm_vector)

    assert not np.isnan(arm_vector).any(), "arm vector has nan"

    forearm_vector = np.array([arm_keypoints[2][0] - arm_keypoints[1][0], arm_keypoints[2][1] - arm_keypoints[1][1], arm_keypoints[2][2] - arm_keypoints[1][2]])
    forearm_vector = __to_unit_vector(forearm_vector)

    assert not np.isnan(forearm_vector).any(), "forearm vector has nan"

    hand_vector = np.array([arm_keypoints[3][0] - arm_keypoints[2][0], arm_keypoints[3][1] - arm_keypoints[2][1], arm_keypoints[3][2] - arm_keypoints[2][2]])
    hand_vector = __to_unit_vector(hand_vector)

    assert not np.isnan(hand_vector).any(), "hand vector has nan"

    # Angle 1
    arm_y_z = np.array([arm_vector[1], arm_vector[2]])
    angle_1 = __angle_between_vectors(vector_1 = arm_y_z, vector_2 = np.array([-1, 0]))

    assert not np.isnan(angle_1).any(), "angle1 have nan"

    # Angle 2
    arm_x_y = np.array([arm_vector[0], arm_vector[1]])
    angle_2 = __angle_between_vectors(vector_1 = arm_y_z, vector_2 = np.array([0, -1]))
    angle_2 = angle_2 * -1

    assert not np.isnan(angle_2).any(), "angle2 have nan"

    # Angle 3
    angle_3 = __angle_between_vectors(vector_1 = arm_vector, vector_2 = forearm_vector)
    angle_3 = angle_3 * -1

    assert not np.isnan(angle_3).any(), "angle3 have nan"

    # Angle 4
    angle_4 = __angle_between_vectors(vector_1 = forearm_vector, vector_2 = hand_vector)
    angle_4 = angle_4 * -1
    
    assert not np.isnan(angle_4).any(), f"angle4 have nan, {angle_4}, {forearm_vector}, {hand_vector}, {forearm_vector[0] == hand_vector[0] and forearm_vector[1] == hand_vector[1] and forearm_vector[2] == hand_vector[2]}"

    return np.array([angle_1, angle_2, angle_3, angle_4])

def __get_angles_from_keypoints(keypoints : np.array):
    assert not np.isnan(keypoints).any(), "nan before alining"
    assert not np.isinf(keypoints).any(), "inf before alining"
    
    keypoints = __align_spine_and_shoulders(keypoints = keypoints)

    assert not np.isnan(keypoints).any(), "nan after alining"
    assert not np.isinf(keypoints).any(), "inf after alining"

    # Right Arm Only
    right_arm_keypoints = np.array([keypoints[2], keypoints[3], keypoints[4], keypoints[5]])

    angles = __get_joint_angles_from_arm_keypoints(arm_keypoints = right_arm_keypoints)

    assert not np.isnan(angles).any(), "nan after arm"
    assert not np.isinf(angles).any(), "inf after arm"

    in_range = True
    for a in angles:
        if not -math.pi <= a <= math.pi:
            in_range = False
            break
    assert in_range, f"{angles} not in range"

    return angles



os.makedirs(OUTPUT_DIR, exist_ok = True)

# Initial Normalization
F = 438
max_x = np.array([-float("inf") for _ in range(F)])
min_x = np.array([float("inf") for _ in range(F)])

print("(1/2) Processing data...")
all_data = []
for motion_file in tqdm(sorted(os.listdir(os.path.join(DATASET_PATH, "motion")))):
    seq_name = motion_file.replace(".json", "")
    motion_path = os.path.join(DATASET_PATH, "motion", f"{seq_name}.json")
    audio_path = os.path.join(DATASET_PATH, "music", f"{seq_name}.wav")

    if seq_name == '000': # This file was bugged for some reason after download
        continue

    keypoints_sequence = smpl_to_3d_keypoints(smpl_json_file = motion_path)
    angles =  np.array([__get_angles_from_keypoints(keypoints = keypoints) for keypoints in keypoints_sequence])
    audio_features = __get_audio_features(audio_path = audio_path)
    
    N = min(angles.shape[0], audio_features.shape[0])
    angles = angles[:N]
    audio_features = audio_features[:N]
        
    assert angles.shape[0] == audio_features.shape[0]
    assert not np.isnan(angles).any(), f"{seq_name} has nan angles"
    assert not np.isinf(angles).any(), f"{seq_name} has inf angles"
    assert not np.isnan(audio_features).any(), f"{seq_name} has nan audio"
    assert not np.isinf(audio_features).any(), f"{seq_name} has inf audio"

    # Store
    all_data.append({
        "file_name" : os.path.join(OUTPUT_DIR, f"{seq_name}.json"),
        "audio_features" : audio_features,
        "joint_angles" : angles
    })

    # Update Normalization
    for x in audio_features:
        min_x = np.min((min_x, x), axis = 0)
        max_x = np.max((max_x, x), axis = 0)

amp_x = max_x - min_x + 1e-9

# Loop Again
print("(2/2) Saving normalized data...")
for sequence_data in tqdm(all_data):
    file_name = sequence_data["file_name"]
    audio_features = sequence_data["audio_features"]
    joint_angles = sequence_data["joint_angles"]

    # Normalize Audio Features
    audio_features = np.array([(x - min_x) / amp_x * 0.8 + 0.1 for x in audio_features])

    assert not np.isnan(audio_features).any(), f"{seq_name} has nan audio"
    assert not np.isinf(audio_features).any(), f"{seq_name} has inf audio"
    assert not np.iscomplex(audio_features).any(), f"{seq_name} has complex audio"

    # Save to file
    with open(file_name, 'w') as f:
        json.dump({
            "audio" : audio_features.tolist(), 
            "angles" : joint_angles.tolist()
        }, f, indent = 4)
