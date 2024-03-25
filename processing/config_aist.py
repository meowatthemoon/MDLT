import os

DATASET_DIR = "<path_to_aist>/aist_plus_plus_dataset"
ANNOTATION_DIR = os.path.join(DATASET_DIR, "annotations")
AUDIO_DIR = os.path.join(DATASET_DIR, "audio")
SMPL_DIR = os.path.join(DATASET_DIR, "models")

FPS = 60
HOP_LENGTH = 512
SR = FPS * HOP_LENGTH

PREPROCESS_OUT_DIR_438 = f"../data_438_{FPS}fps"
PREPROCESS_OUT_DIR_35 = f"../data_35_{FPS}fps"
NORMALIZATION_438 = "normalization_438.json"
NORMALIZATION_35 = "normalization_35.json"
