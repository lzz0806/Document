import gym_super_mario_bros
from gym.spaces import Box
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import cv2
import numpy as np





def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))


class CustomReward(Wrapper):
    def __init__(self, env=None, world=None, stage=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.num_actions =7
        self.curr_score = 0
        self.current_x = 40
        self.world = world
        self.stage = stage

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = process_frame(state)
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        if self.world == 7 and self.stage == 4:
            if (506 <= info["x_pos"] <= 832 and info["y_pos"] > 127) or (
                    832 < info["x_pos"] <= 1064 and info["y_pos"] < 80) or (
                    1113 < info["x_pos"] <= 1464 and info["y_pos"] < 191) or (
                    1579 < info["x_pos"] <= 1943 and info["y_pos"] < 191) or (
                    1946 < info["x_pos"] <= 1964 and info["y_pos"] >= 191) or (
                    1984 < info["x_pos"] <= 2060 and (info["y_pos"] >= 191 or info["y_pos"] < 127)) or (
                    2114 < info["x_pos"] < 2440 and info["y_pos"] < 191) or info["x_pos"] < self.current_x - 500:
                reward -= 50
                done = True
        if self.world == 4 and self.stage == 4:
            if (info["x_pos"] <= 1500 and info["y_pos"] < 127) or (
                    1588 <= info["x_pos"] < 2380 and info["y_pos"] >= 127):
                reward = -50
                done = True

        self.current_x = info["x_pos"]
        return state, reward / 10., done, info

    def reset(self):
        self.curr_score = 0
        self.current_x = 40
        return process_frame(self.env.reset())


class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.num_states = self.observation_space.shape[0]
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)

    def step(self, action):
        total_reward = 0
        last_states = []
        for i in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if i >= self.skip / 2:
                last_states.append(state)
            if done:
                self.reset()
                return self.states[None, :, :, :].astype(np.float32), total_reward, done, info
        max_state = np.max(np.concatenate(last_states, 0), 0)
        self.states[:-1] = self.states[1:]
        self.states[-1] = max_state
        return self.states[None, :, :, :].astype(np.float32), total_reward, done, info

    def reset(self):
        state = self.env.reset()
        self.states = np.concatenate([state for _ in range(self.skip)], 0)
        return self.states[None, :, :, :].astype(np.float32)


def create_train_env(world, stage, actions, output_path=None):
    if actions == "right":
        actions = RIGHT_ONLY
    elif actions == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))
    env = JoypadSpace(env, actions)
    env = CustomReward(env, world, stage)
    env = CustomSkipFrame(env)
    return env
