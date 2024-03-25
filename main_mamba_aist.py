import argparse
import json
import os

import numpy as np

from train_mamba import experiment

ALL_GENRES = ['mPO', 'mLO', 'mWA', 'mJB', 'mLH', 'mMH', 'mBR', 'mKR', 'mHO', 'mJS']
ALL_INDEXES = [0, 1, 2, 3, 4, 5]
DEMONSTRATION_DATA = './aistpp_data_438_60fps'

def load_demonstrations(dataset_path : str, music_pieces : list):
    state_sequences, action_sequences, max_ep_len = [], [], 0
    for demonstration_file in os.listdir(dataset_path):
        demonstration_music_piece = demonstration_file.split("_")[-2]

        if demonstration_music_piece not in music_pieces:
            continue

        with open(os.path.join(dataset_path, demonstration_file), "r") as f:
            demonstration_data = json.load(f)
            state_sequence = demonstration_data["audio"]
            action_sequence = demonstration_data["angles"]

            assert len(state_sequence) == len(action_sequence)

            sequence_length = len(state_sequence)
            max_ep_len = max(max_ep_len, sequence_length)

            state_sequences.append(np.array(state_sequence))
            action_sequences.append(np.array(action_sequence))

    return state_sequences, action_sequences, max_ep_len

def main(genre : str, val_index : int, infer_every : int, n_epochs : int, sequence_length : int, n_layer : int, d_model : int):
    assert genre == 'all' or genre in ALL_GENRES, f"Invalid genre : {genre}. Use 'all' or {ALL_GENRES}."
    assert val_index == -1 or val_index in ALL_INDEXES, f"Invalid val_index : {val_index}. Use '-1' or {ALL_INDEXES}."
    assert sequence_length is not None, "Sequence length 'K' can't be None."
    assert d_model is not None, "d_model can't be None."
    assert n_layer is not None, "n_layer can't be None."

    experiment_name = f'E{n_epochs}_H{n_layer}_D{d_model}_K{sequence_length}_genre_{genre}_val_{val_index if val_index != -1 else "all"}'

    exp_genres = ALL_GENRES if genre == 'all' else [genre]
    train_indexes = ALL_INDEXES if val_index == -1 else [i for i in ALL_INDEXES if i != val_index]
    val_indexes = [val_index] if val_index != -1 else ALL_INDEXES

    train_pieces = [f"{exp_genre}{train_index}" for exp_genre in exp_genres for train_index in train_indexes]
    val_pieces = [f"{exp_genre}{val_index}" for exp_genre in exp_genres for val_index in val_indexes]

    train_state_sequences, train_action_sequences, train_max_ep_len = load_demonstrations(dataset_path = DEMONSTRATION_DATA, music_pieces = train_pieces)
    val_state_sequences, val_action_sequences, val_max_ep_len = load_demonstrations(dataset_path = DEMONSTRATION_DATA, music_pieces = val_pieces)
    max_ep_len = max(train_max_ep_len, val_max_ep_len)

    experiment(
        exp_name = experiment_name, 
        train_state_sequences = train_state_sequences, 
        train_action_sequences = train_action_sequences, 
        val_state_sequences = val_state_sequences, 
        val_action_sequences = val_action_sequences, 
        max_ep_len = max_ep_len,
        infer_every = infer_every,
        n_epochs = n_epochs,
        sequence_length = sequence_length,
        d_model = d_model,
        n_layer = n_layer
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--genre", type = str)
    parser.add_argument("--val_index", type = int)
    parser.add_argument("--infer_every", type = int)
    parser.add_argument("--K", type = int)
    parser.add_argument("--d_model", type = int)
    parser.add_argument("--n_layer", type = int)
    parser.add_argument("--n_epochs", type = int, default = 1000000)
    args = parser.parse_args()

    main(genre = args.genre, val_index = args.val_index, infer_every = args.infer_every, n_epochs = args.n_epochs, sequence_length = args.K, d_model = args.d_model, n_layer = args.n_layer)
