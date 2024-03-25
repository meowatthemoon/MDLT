import json
import math
import os

import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import Adam

from mamba_memory import Memory
from metrics import rad_mse, calc_fid
from models.mamba import MambaLMHeadModel

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

def save_model(model : MambaLMHeadModel, save_path : str):
    torch.save(model.state_dict(), save_path)

def inference_old(model : MambaLMHeadModel, state_sequence : np.array, state_size : int, action_size : int, action_range : float, sequence_length : int, device : torch.device, dtype) -> np.array:
    model = model.eval()
    model.to(device)

    with torch.no_grad():
        state_tensor = torch.tensor(state_sequence, device = device, dtype = dtype).unsqueeze(0)# (1, Music Length, state_size)
        
        action_preds = model(state_tensor) * action_range # (1, Music Length, action_size)

        action_tensor = action_preds.squeeze(0) # (Music Length, action_size)

        return action_tensor.cpu().numpy()
    
def inference(model : MambaLMHeadModel, state_sequence : np.array, state_size : int, action_size : int, action_range : float, sequence_length : int, device : torch.device, dtype) -> np.array:
    model = model.eval()
    model.to(device)

    states = np.zeros((0, state_size), dtype = np.float32)

    new_action_sequence = [] # This is the output, where the generated actions are stored

    with torch.no_grad():
        for i, state in enumerate(state_sequence):
            states = np.concatenate([states, state.reshape(1, state_size)])

            L = states.shape[0]
            if L > sequence_length:
                states = states[1:]

            state_tensor = torch.tensor(states, device = device, dtype = torch.float32).reshape(1, -1, state_size)

            action_preds = model(state_tensor)
            action = action_preds[:, -1].cpu().numpy() * action_range

            action = model(state_tensor)[:, -1].cpu().numpy() * action_range
            new_action_sequence.append(action[0])

    return np.array(new_action_sequence)

def experiment(exp_name : str, train_state_sequences : list, train_action_sequences : list, val_state_sequences : list, val_action_sequences : list, max_ep_len : int, sequence_length : int, d_model : int, n_layer : int, infer_every : int = 5000, n_epochs : int = 1000000):
    # Parameters
    action_range : float = math.pi
    action_size : int = 4
    state_size : int = 438

    batch_size : int = 16
    dtype = torch.float32#torch.float16
    lr : float = 1e-4

    device : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dirs
    model_path = f"results/{exp_name}"
    os.makedirs(model_path, exist_ok = True)  

    # Create Memory
    memory = Memory(max_trajectories = len(train_state_sequences), max_episode_length = max_ep_len, state_size = state_size, action_size = action_size, device = device, dtype = dtype)

    # Load demonstrations into memory
    for (state_sequence, action_sequence) in zip(train_state_sequences, train_action_sequences):
        memory.start_new_trajectory()
        for t_i in range(len(state_sequence)):
            memory.store_transition(state = state_sequence[t_i], action = action_sequence[t_i])
    print(f"Loaded {len(train_state_sequences)} train demonstrations, with a maximum episode length of {max_ep_len}.")

    # Create Model
    model = MambaLMHeadModel(state_size = state_size, action_size = action_size, d_model = d_model, n_layer = n_layer, device = device, dtype = dtype)
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

        encoder_input, label, encoder_mask = memory.sample_buffer(batch_size = batch_size, sequence_length = sequence_length)

        proj_output = model(encoder_input) # (Batch, Seq_len, action_size)

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
                new_action_sequence = inference_old(model = model, state_sequence = state_sequence, state_size = state_size, action_size = action_size, action_range = action_range, sequence_length = sequence_length, device = device, dtype = dtype)
                
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
