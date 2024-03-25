import json
import os

import essentia
#import essentia.streaming
from essentia.standard import *
import numpy as np
from tqdm import tqdm

from extractor import FeatureExtractor
from config_aist import ANNOTATION_DIR, AUDIO_DIR, NORMALIZATION_438, PREPROCESS_OUT_DIR_438, SR, FPS
from general_aist import get_arm_joint_angles_from_sequence, load_timestamps, synchronize

def __get_audio_features(sequence_name : str) -> np.array:
    audio_name = sequence_name.split("_")[-2]
    audio_file = os.path.join(AUDIO_DIR, f"{audio_name}.wav")
    loader = None
    try:
        loader = essentia.standard.MonoLoader(filename = audio_file, sampleRate = SR)
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
    chroma_cqt = extractor.get_chroma_cqt(audio_harmonic, SR, octave=7 if SR ==15360*2 else 5)
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

def process_aist():
    # Initial Normalization
    F = 438
    max_x = np.array([-float("inf") for _ in range(F)])
    min_x = np.array([float("inf") for _ in range(F)])

    # Load Sequences   
    sequences = []
    for split in ["train", "val", "test"]:
        new_sequences = np.loadtxt(os.path.join(ANNOTATION_DIR, f"splits/crossmodal_{split}.txt"), dtype=str).tolist()
        sequences += new_sequences
    
    # Process    
    ignore_sequences = np.loadtxt(os.path.join(ANNOTATION_DIR, "ignore_list.txt"), dtype=str).tolist()
    sequences = [name for name in sequences if name not in ignore_sequences]
    sequences = sorted(sequences)

    output_dir = os.path.join(PREPROCESS_OUT_DIR_438)
    os.makedirs(output_dir, exist_ok = True)

    all_data = []
    done_music = []

    for sequence in tqdm(sequences):
        music = sequence.split("_")[-2]

        if music in done_music:
            continue
        done_music.append(music)

        # Keypoints
        joint_angles = get_arm_joint_angles_from_sequence(sequence_name = sequence) # N1 x 4, N1 = number of annotations
        
        # Timestamps
        timestamps_micro = load_timestamps(sequence_name = sequence) # N1, N1 = number of annotations

        # Audio Features
        audio_features = __get_audio_features(sequence_name = sequence) # N2 x F, N2 = length of sequence, F = number of features

        # Synchronize
        audio_features, joint_angles, timestamps_micro = synchronize(audio_features = audio_features, joint_angles = joint_angles, timestamps_micro = timestamps_micro, fps = FPS)

        # Joint Angles : N2 x 4
        # Timestamps : N2
        # Audio Feautes : N2 x F
        
        # Store
        all_data.append({
            "file_name" : os.path.join(output_dir, f"{sequence}.json"),
            "audio_features" : audio_features,
            "joint_angles" : joint_angles#,
            #"timestamps_micro" : timestamps_micro
        })
        
        # Update Normalization
        for x in audio_features:
            min_x = np.min((min_x, x), axis = 0)
            max_x = np.max((max_x, x), axis = 0)


    # Save Normalization
    amp_x = max_x - min_x + 1e-9
    with open(NORMALIZATION_438, 'w') as f:
        json.dump({"min_x" : min_x.tolist(), "max_x" : max_x.tolist()}, f, indent = 4)

    # Loop Again
    for sequence_data in all_data:
        file_name = sequence_data["file_name"]
        audio_features = sequence_data["audio_features"]
        joint_angles = sequence_data["joint_angles"]
        #timestamps_micro = sequence_data["timestamps_micro"]

        # Normalize Audio Features
        audio_features = np.array([(x - min_x) / amp_x * 0.8 + 0.1 for x in audio_features])

        # Save to file
        with open(file_name, 'w') as f:
            json.dump({
                "audio" : audio_features.tolist(), 
                "angles" : joint_angles.tolist()#, 
                #"timestamps" : timestamps_micro.tolist()
            }, f, indent = 4)

if __name__ == '__main__':
    process_aist()
