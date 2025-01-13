import numpy as np
import torch
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import warnings
from gym.spaces import Box
import matplotlib.pyplot as plt
from sympy.stats.sampling.sample_numpy import numpy

warnings.filterwarnings("ignore")


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
import torch.nn as nn
import torch.nn.functional as F
import cv2


def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))
        frame = frame[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))

class PPO(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(PPO, self).__init__()
        print(num_inputs, num_actions)
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                # nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.linear(x)
        return self.actor_linear(x), self.critic_linear(x)


state = env.reset()
state = process_frame(state)
state = np.concatenate([state for _ in range(4)], 0)
print('xxx', state.shape)
state = state[None, :, :, :].astype(np.float32)
print('lzz', state.shape)
state = torch.tensor(np.array(state), dtype=torch.float32)
obs_shape = Box(low=0, high=255, shape=(4, 84, 84))
print('2222', state.shape)
print("ssss", obs_shape.shape[0])
ppo =  PPO(obs_shape.shape[0], 7)
actor, critic = ppo(state)
print("actor", actor)
print("critic", critic)
# print(state.shape)

