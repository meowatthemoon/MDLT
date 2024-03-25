import random

import numpy as np
import torch

class Memory:
    def __init__(self, max_trajectories : int, max_episode_length : int, state_size : int, action_size : int, device : torch.device, dtype):
        self.max_trajectories : int = max_trajectories
        self.trajectory_counter : int = -1

        self.state_size : int = state_size
        self.action_size : int = action_size
        self.max_episode_length : int = max_episode_length
        self.device : torch.device = device

        self.dtype = dtype

        self.length_memory : np.array = np.zeros((self.max_trajectories), dtype = np.int64)
        self.state_memory : np.array = np.zeros((self.max_trajectories, self.max_episode_length, self.state_size), dtype = np.float32)
        self.action_memory : np.array = np.zeros((self.max_trajectories, self.max_episode_length, self.action_size), dtype = np.float32)

    def get_vocabulary_size(self) -> int:
        return self.max_episode_length
    
    def start_new_trajectory(self):
        self.trajectory_counter += 1
        self.trajectory_counter = self.trajectory_counter % self.max_trajectories

        self.length_memory[self.trajectory_counter] = 0
        self.state_memory[self.trajectory_counter] = np.zeros((self.max_episode_length, self.state_size))
        self.action_memory[self.trajectory_counter] = np.zeros((self.max_episode_length, self.action_size))

    def store_transition(self, state : np.array, action : np.array):
        index = self.length_memory[self.trajectory_counter]

        self.state_memory[self.trajectory_counter][index] = state
        self.action_memory[self.trajectory_counter][index] = action
        self.length_memory[self.trajectory_counter] += 1

    def sample_buffer(self, batch_size : int, sequence_length : int):
        trajectory_indices = np.random.choice(
            np.arange(self.max_trajectories),
            size = batch_size,
            replace = True,
            p = None,
        )

        encoder_input_batch = np.zeros((batch_size, sequence_length, self.state_size))
        decoder_input_batch = np.zeros((batch_size, sequence_length, self.action_size))
        encoder_mask_batch = np.zeros((batch_size, 1, 1, sequence_length))

        for i, trajectory_index in enumerate(trajectory_indices):
            states, actions, encoder_mask = self.__sample_trajectory(trajectory_index = trajectory_index, sequence_length = sequence_length)

            encoder_input_batch[i] = states
            decoder_input_batch[i] = actions
            encoder_mask_batch[i] = encoder_mask

        encoder_input_tensor = torch.tensor(encoder_input_batch, device = self.device, dtype = self.dtype) # Batch_Size, Sequence_Length, State_Size 
        decoder_input_tensor = torch.tensor(decoder_input_batch, device = self.device, dtype = self.dtype) # Batch_Size, Sequence_Length, Action_Size
        encoder_mask_tensor = torch.tensor(encoder_mask_batch, device = self.device, dtype = torch.int) # Batch_Size, 1, 1, Sequence_Length

        return encoder_input_tensor, decoder_input_tensor, encoder_mask_tensor
    
    def __sample_trajectory(self, trajectory_index : int, sequence_length : int):
        trajectory_length = self.length_memory[trajectory_index]

        states = self.state_memory[trajectory_index]
        actions = self.action_memory[trajectory_index]
        timesteps = np.arange(1, trajectory_length + 1)

        si = random.randint(0, trajectory_length - 1)
        sf = si + 1

        states = self.state_memory[trajectory_index][si : sf]
        actions = self.action_memory[trajectory_index][si : sf]

        # Padding
        sub_length = states.shape[0]
        padding_size = sequence_length - sub_length

        states = np.concatenate([states, np.zeros((padding_size, self.state_size))]) # Sequence_Length, State_Size
        actions = np.concatenate([actions, np.zeros((padding_size, self.action_size))]) # Sequence_Length, Action_Size
        encoder_mask = np.concatenate([np.ones(sub_length), np.zeros(padding_size)])

        return states, actions, encoder_mask
