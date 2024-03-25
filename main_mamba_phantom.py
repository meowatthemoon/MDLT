import argparse
import json
import os

import numpy as np

from train_mamba import experiment

DEMONSTRATION_DATA = './phantom_data_438_30fps'

def load_demonstrations(dataset_path : str, val_type : int):
    train_state_sequences, train_action_sequences, val_state_sequences, val_action_sequences, max_ep_len = [], [], [], [], 0
    N = len(os.listdir(dataset_path))
    for i, demonstration_file in enumerate(sorted(os.listdir(dataset_path))):

        with open(os.path.join(dataset_path, demonstration_file), "r") as f:
            demonstration_data = json.load(f)
            state_sequence = demonstration_data["audio"]
            action_sequence = demonstration_data["angles"]

            assert len(state_sequence) == len(action_sequence), f"{demonstration_file} not same length..."
            
            """
            assert not np.isnan(state_sequence).any(), f"{demonstration_file} has nan states..."
            assert not np.isinf(state_sequence).any(), f"{demonstration_file} has inf states..."
            assert not np.isnan(action_sequence).any(), f"{demonstration_file} has nan actions..."
            assert not np.isinf(action_sequence).any(), f"{demonstration_file} has inf actions..."
            """
            if np.isnan(state_sequence).any() or np.isinf(state_sequence).any() or np.isnan(action_sequence).any() or np.isinf(action_sequence).any():
                print(f"Skipping {demonstration_file} due to invalid values")
                continue


            sequence_length = len(state_sequence)
            max_ep_len = max(max_ep_len, sequence_length)

            if i + 1 >= N * val_type/10 and i + 1 < N * (val_type + 1)/10:
                val_state_sequence = np.array(state_sequence)
                val_action_sequence = np.array(action_sequence)
                val_state_sequences.append(val_state_sequence)
                val_action_sequences.append(val_action_sequence)
            else:
                train_state_sequence = np.array(state_sequence)
                train_action_sequence = np.array(action_sequence)
                train_state_sequences.append(train_state_sequence)
                train_action_sequences.append(train_action_sequence)              

    return train_state_sequences, train_action_sequences, val_state_sequences, val_action_sequences, max_ep_len

def main(val_type : int, infer_every : int, n_epochs : int, sequence_length : int, n_layer : int, d_model : int):
    assert 0 < val_type < 9 , f"Invalid val_type : {val_type}. Use [0, ..., 9]."
    assert sequence_length is not None, "Sequence length 'K' can't be None."
    assert d_model is not None, "d_model can't be None."
    assert n_layer is not None, "n_layer can't be None."

    experiment_name = f'PHANTOM2_E{n_epochs}_H{n_layer}_D{d_model}_K{sequence_length}_val_{val_type}'

    train_state_sequences, train_action_sequences, val_state_sequences, val_action_sequences, max_ep_len = load_demonstrations(dataset_path = DEMONSTRATION_DATA, val_type = val_type)

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
    parser.add_argument("--val_type", type = int)
    parser.add_argument("--infer_every", type = int)
    parser.add_argument("--K", type = int)
    parser.add_argument("--d_model", type = int)
    parser.add_argument("--n_layer", type = int)
    parser.add_argument("--n_epochs", type = int, default = 250000)
    args = parser.parse_args()

    main(val_type = args.val_type, infer_every = args.infer_every, n_epochs = args.n_epochs, sequence_length = args.K, d_model = args.d_model, n_layer = args.n_layer)
