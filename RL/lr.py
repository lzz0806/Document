import numpy as np
import torch
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import warnings
from gym.spaces import Box
import matplotlib.pyplot as plt
from sympy.physics.units import femto
from sympy.stats.sampling.sample_numpy import numpy

warnings.filterwarnings("ignore")


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# x = torch.randn(2, 3, 4)
# print(x)
# new_x = x.view(x.size(1), -1)
# print(new_x.shape)
import cv2

def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))
        frame = frame[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))

# state = env.reset()
# b = np.array(state)
# print(b.shape)
# val = process_frame(state)
# print(val.shape)
# observation_space = Box(low=0, high=255, shape=(1, 84, 84))
# print(observation_space.shape[0])
action = torch.tensor(4)
print(action)
action = SIMPLE_MOVEMENT[action]
print(action)
