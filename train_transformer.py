import json
import math
import os

import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import Adam

from transformer_memory import Memory, causal_mask
from metrics import rad_mse, calc_fid
from models.transformer import build_transfomer, Transformer


def inference(model : Transformer, state_sequence : np.array, state_size : int, action_size : int, action_range : float, sequence_length : int, device : torch.device) -> np.array:
    model = model.eval()
    model.to(device)
    soa = np.zeros((1, action_size), dtype = np.float32)

    states = np.zeros((0, state_size), dtype = np.float32) # (0, state_size)
    actions = np.concatenate([soa], axis = 0) # (1, action_size) 
    timesteps = np.zeros((0,), dtype = np.int64) # (1,)

    new_action_sequence = [] # This is the output, where the generated actions are stored

    with torch.no_grad():
        for i, state in enumerate(state_sequence):
            index = i + 1
            state : np.array
            
            timesteps = np.concatenate([timesteps, np.array([index])])
            states = np.concatenate([states, state.reshape(1, state_size)])

            L = states.shape[0]
            if L > sequence_length:
                states = states[1:]
                timesteps = timesteps[1:]
            
            encoder_mask = np.ones(states.shape[0])  
            decoder_mask = causal_mask(size = actions.shape[0]).numpy()
            encoder_mask = encoder_mask.reshape(1, 1, states.shape[0])
            
            state_tensor = torch.tensor(states, device = device, dtype = torch.float32).reshape(1, -1, state_size)
            timestep_tensor = torch.tensor(timesteps, device = device, dtype = torch.int64).reshape(1, -1)
            
            action_tensor = torch.tensor(actions, device = device, dtype = torch.float32).reshape(1, -1, action_size)
            encoder_mask_tensor = torch.tensor(encoder_mask, device = device, dtype = torch.int).reshape(1, 1, 1, -1)
            decoder_mask_tensor = torch.tensor(decoder_mask, device = device, dtype = torch.int).reshape(1, 1, actions.shape[0], actions.shape[0])

            encoder_output = model.encode(state = state_tensor, timestep = timestep_tensor, src_mask = encoder_mask_tensor)
            decoder_output = model.decode(encoder_output = encoder_output, timestep = timestep_tensor, src_mask = encoder_mask_tensor, action = action_tensor, tgt_mask = decoder_mask_tensor)
            action = model.project(decoder_output[:, -1]).cpu().numpy() * action_range
            actions = np.concatenate([actions, action])

            L = actions.shape[0]
            if L > sequence_length:
                actions = actions[1:]

            # Store in output
            new_action_sequence.append(action[0])

    return np.array(new_action_sequence)

def save_results(results_path : str, train_losses : list, val_mse_errors : list, validation_fids : list):
    with open(os.path.join(results_path), "w") as f:
        json.dump({
            "train_losses" : train_losses,
            "validation_errors_radians" : val_mse_errors,
            "validation_fids" : validation_fids,
            "lowest_mse" : np.min(val_mse_errors),
            "lowest_fid" : np.min(validation_fids),
            "highest_fid" : np.max(validation_fids)
        },fp = f, indent = 4)

def save_model(model : Transformer, save_path : str):
    torch.save(model.state_dict(), save_path)

def experiment(exp_name : str, train_state_sequences : list, train_action_sequences : list, val_state_sequences : list, val_action_sequences : list, max_ep_len : int, sequence_length : int, infer_every : int = 5000, n_epochs : int = 1000000, d_model : int = 512, n_layers : int = 6):
    # Parameters
    action_range : float = math.pi
    action_size : int = 4
    state_size : int = 438

    batch_size : int = 16
    lr : float = 1e-4

    device : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dirs
    model_path = f"results/{exp_name}"
    os.makedirs(model_path, exist_ok = True)    
    
    # Create Memory
    memory = Memory(max_trajectories = len(train_state_sequences), max_episode_length = max_ep_len, state_size = state_size, action_size = action_size, device = device)
    vocabulary_size : int = memory.get_vocabulary_size()

    # Load demonstrations into memory
    for (state_sequence, action_sequence) in zip(train_state_sequences, train_action_sequences):
        memory.start_new_trajectory()
        for t_i in range(len(state_sequence)):
            memory.store_transition(state = state_sequence[t_i], action = action_sequence[t_i])
    print(f"Loaded {len(train_state_sequences)} train demonstrations, with a maximum episode length of {max_ep_len}.")

    # Create model
    model = build_transfomer(state_size = state_size, action_size = action_size, max_ep_len = vocabulary_size, d_model = d_model, n_layers = n_layers)
    model = model.to(device = device)

    # Optimizer
    optimizer = Adam(model.parameters(), lr = lr)

    # Loss Function
    loss_fn = MSELoss(reduction = 'none').to(device = device)

    # Train loop
    train_losses, val_mse_errors, val_fid_errors = [], [], []
    best_mse_error = float("inf")
    lowest_fid_error = float("inf")
    for epoch in range(n_epochs):
        model = model.train()
        encoder_input, decoder_input, timestep, label, encoder_mask, decoder_mask = memory.sample_buffer(batch_size = batch_size, sequence_length = sequence_length)
        encoder_output = model.encode(state = encoder_input, timestep = timestep, src_mask = encoder_mask) # (Batch, Seq_len, d_model)
        decoder_output = model.decode(encoder_output = encoder_output, src_mask = encoder_mask, action = decoder_input, timestep = timestep, tgt_mask = decoder_mask) # (Batch, Seq_len, d_model)
        proj_output = model.project(x = decoder_output) # (Batch, Seq_len, action_size)
        scaled_output = proj_output * action_range # Model outputs tanh [-1, 1], scale it to [-action_range, action_range]

        loss = loss_fn(scaled_output, label) # (Batch, Seq_len, action_size)
        loss_mask = encoder_mask.reshape(encoder_mask.shape[0], encoder_mask.shape[-1]) # (Batch, Seq_len)
        loss = loss[loss_mask > 0].mean() # (Batch, Seq_len)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_losses.append(loss.item())

        # Inference
        if (epoch + 1 ) % infer_every == 0:
            radian_errors, fid_errors = [], []
            for (state_sequence, action_sequence) in zip(val_state_sequences, val_action_sequences):
                new_action_sequence = inference(model = model, state_sequence = state_sequence, state_size = state_size, action_size = action_size, action_range = action_range, sequence_length = sequence_length, device = device)
                
                mse_error = rad_mse(new_action_sequence, action_sequence)
                radian_errors.append(mse_error)

                fid_error = calc_fid(new_action_sequence, action_sequence) # Order of fid does not matter, outputs the same
                fid_errors.append(fid_error)
            
            avg_mse_error = np.mean(radian_errors)
            val_mse_errors.append(avg_mse_error)
            avg_fid_error = np.mean(fid_errors)
            val_fid_errors.append(avg_fid_error)

            # Save best
            best_mse_error = min(best_mse_error, avg_mse_error)
            lowest_fid_error = min(lowest_fid_error, avg_fid_error)
            save_results(results_path = os.path.join(model_path, "results.json"), train_losses = train_losses, val_mse_errors = val_mse_errors, validation_fids = val_fid_errors)
            print(f"{exp_name} | {epoch + 1} / {n_epochs} | Avg mse : {avg_mse_error:.3f} radians | Best mse : {best_mse_error:.3f} | Avg fid : {avg_fid_error:.2f} | Lowest fid : {lowest_fid_error:.2f}")

    # Save final
    save_results(results_path = os.path.join(model_path, "results.json"), train_losses = train_losses, val_mse_errors = val_mse_errors, validation_fids = val_fid_errors)
    
