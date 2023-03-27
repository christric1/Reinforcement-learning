import random
import numpy as np
from collections import namedtuple
from nets.dqn import dqn

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn

# Transition
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyperparameter
number_of_actions = 6
actions_of_history = 4
reward_movement_action = 1
reward_terminal_action = 3
iou_threshold = 0.5

EPSILON = 0.5
GAMMA = 0.90

transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))  #  numbers here need to be adjusted in future
])


class DQNAgent:
    def __init__(self, obs_dim, action_dim, batch_size=16, device='cuda'):
        self.memory = ReplayMemory(1000)
        self.batch_size = batch_size
        
        # networks: dqn, dqn_target
        self.dqn = dqn(obs_dim, action_dim).to(device)
        self.dqn_target = dqn(obs_dim, action_dim).to(device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())
        self.criterion = nn.MSELoss()

    def train(self):
        self.dqn.train()

    def eval(self):
        self.dqn.eval()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        if random.random() < EPSILON:
            action = torch.randint(0, 6, size=(1,)).squeeze()
        else:
            qval = self.dqn(state)
            _, predicted = torch.max(qval.data, 1)
            action = predicted[0]
        return action.item()

    def update_model(self) -> torch.Tensor:
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor(batch.action).view(self.batch_size, -1)
        reward_batch = torch.tensor(batch.reward)

        # torch.Size([16, 1])
        state_action_values = self.dqn(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.dqn_target(non_final_next_states).max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute loss
        loss = self.criterion(state_action_values.squeeze(), expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
       
def get_state(image, history, backbone, device):
    im = transform(image[0]).unsqueeze(dim=0).to(device)
    feature = backbone(im).view(1, -1)
    history_vector = history.view(1, -1)
    state = torch.cat((feature, history_vector), 1)
    return state

def update_history_vector(history_vector, action):
    action_vector = torch.zeros(number_of_actions)
    action_vector[action-1] = 1
    size_history_vector = len(torch.nonzero(history_vector))   

    if size_history_vector < actions_of_history:
        history_vector[size_history_vector][action-1] = 1
    else:
        for i in range(actions_of_history-1, 0, -1):
            history_vector[i][:] = history_vector[i-1][:]
        history_vector[0][:] = action_vector[:] 

    return history_vector

def get_reward_movement(iou, new_iou):
    if new_iou > iou:
        reward = reward_movement_action
    else:
        reward = - reward_movement_action
    return reward

def get_reward_trigger(new_iou):
    if new_iou > iou_threshold:
        reward = reward_terminal_action
    else:
        reward = - reward_terminal_action
    return reward