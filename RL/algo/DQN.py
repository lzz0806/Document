import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from RL.config import AlgoConfig, EnvConfig
from RL.replay_buffer import ReplayBuffer
from RL.rl_nn import MLP


class DQN:

     def __init__(self, cfg: AlgoConfig, env_cfg: EnvConfig, memory: ReplayBuffer):
          self.config = cfg
          self.env_config = env_cfg
          self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
          # env 参数
          self.actions_dim = self.env_config.actions_dim
          # e-greedy 参数
          self.epsilon = 0
          self.sample_count = 0
          self.epsilon_start = cfg['epsilon_start']
          self.epsilon_end = cfg['epsilon_end']
          self.epsilon_decay = cfg['epsilon_decay']
          self.batch_size = cfg['batch_size']
          self.policy_net = MLP().to(self.device)
          self.target_net = MLP().to(self.device)
          
          # 将策略网络的参数copy到目标网络
          for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
               target_param.data.copy_(param.data)

          self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
          self.loss_fn = nn.MSELoss()
          self.memory = memory
     
     def chooise_action(self, state):
          self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon) * math.exp(-self.sample_count / self.epsilon_decay)
          if random.random() > self.epsilon:
               # 如果探索概率大于 【】，就使用 policy_net 进行 action 选择
               with torch.no_grad():
                    state = torch.tensor(state, device=self.device, detype=torch.float32).unsqueeze(0)
                    print('单步状态为', state)
                    q_values = self.policy_net(state)
                    action = q_values.max(1)[1].item()
          else:
               action = random.randrange(self.actions_dim)
          return action

     def predict_action(self, state):
          # 预测动作
          state = torch.tensor(state, device=self.device, detype=torch.float32).unsqueeze(0)
          with torch.no_grad():
               q_values = self.policy_net(state)
               action = q_values.max(1)[1].item()
          return action

     def update(self):
          if len(self.memory) < self.batch_size:
               return
          # 随机获取 batch_size 个 experience
          batch = self.memory.sample(self.batch_size)
          state_batch = torch.tensor([e[0] for e in batch], device=self.device, dtype=torch.float32)
          action_batch = torch.tensor([e[1] for e in batch], device=self.device, dtype=torch.long)
          reward_batch = torch.tensor([e[2] for e in batch], device=self.device, dtype=torch.float32)
          next_state_batch = torch.tensor([e[3] for e in batch], device=self.device, dtype=torch.float32)
          done_batch = torch.tensor([e[4] for e in batch], device=self.device, dtype=torch.bool)
          # 计算Q值和下一个step的Q值
          q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch) # 计算当前状态(s_t,a)对应的Q(s_t, a)
          next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
          # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
          expected_q_values = reward_batch + self.gamma * next_q_values * (1-done_batch)
          loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))  # 计算均方根损失
          # 优化更新模型
          self.optimizer.zero_grad()  
          loss.backward()
          # clip防止梯度爆炸
          for param in self.policy_net.parameters():  
               param.grad.data.clamp_(-1, 1)
          self.optimizer.step() 

def train(cfg, env, agent):
    ''' 训练
    '''
    print("开始训练！")
    rewards = []  # 记录所有回合的奖励
    steps = []
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.sample_action(state)  # 选择动作
            next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
            agent.memory.push((state, action, reward,next_state, done))  # 保存transition
            state = next_state  # 更新下一个状态
            agent.update()  # 更新智能体
            ep_reward += reward  # 累加奖励
            if done:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            print(f"回合：{i_ep+1}/{cfg.train_eps}，奖励：{ep_reward:.2f}，Epislon：{agent.epsilon:.3f}")
    print("完成训练！")
    env.close()
    return {'rewards':rewards}

def test(cfg, env, agent):
    print("开始测试！")
    rewards = []  # 记录所有回合的奖励
    steps = []
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # 记录一回合内的奖励
        state = env.reset()  # 重置环境，返回初始状态
        for _ in range(cfg.max_steps):
            action = agent.predict_action(state)  # 选择动作
            next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            if done:
                break
        rewards.append(ep_reward)
        print(f"回合：{i_ep+1}/{cfg.test_eps}，奖励：{ep_reward:.2f}")
    print("完成测试")
    env.close()
    return {'rewards':rewards}

import gym
import os
def all_seed(env,seed = 1):
    ''' 万能的seed函数
    '''
    # env.seed(seed) # env config
    # np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # config for CPU
    torch.cuda.manual_seed(seed) # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # config for python scripts
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
def env_agent_config(cfg):
    env = gym.make(cfg.env_name) # 创建环境
    all_seed(env,seed=cfg.seed)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print(f"状态空间维度：{n_states}，动作空间维度：{n_actions}")
    # 更新n_states和n_actions到cfg参数中
    setattr(cfg, 'n_states', n_states)
    setattr(cfg, 'n_actions', n_actions) 
    agent = DQN(cfg)
    return env,agent

import matplotlib.pyplot as plt
class Config:
    def __init__(self):
        self.algo_name = 'DoubleDQN' # 算法名称
        self.env_name = 'CartPole-v1' # 环境名称
        self.seed = 1 # 随机种子
        self.train_eps = 100 # 训练回合数
        self.test_eps = 10  # 测试回合数
        self.max_steps = 200 # 每回合最大步数
        self.gamma = 0.99 # 折扣因子
        self.lr = 0.0001 # 学习率
        self.epsilon_start = 0.95 # epsilon初始值
        self.epsilon_end = 0.01 # epsilon最终值
        self.epsilon_decay = 500 # epsilon衰减率
        self.buffer_size = 10000 # ReplayBuffer容量
        self.batch_size = 64 # ReplayBuffer中批次大小
        self.target_update = 4 # 目标网络更新频率
        self.hidden_dim = 256 # 神经网络隐藏层维度
        if torch.cuda.is_available(): # 是否使用GPUs
            self.device = 'cuda'
        else:
            self.device = 'cpu'
def smooth(data, weight=0.9):  
    '''用于平滑曲线，类似于Tensorboard中的smooth曲线
    '''
    last = data[0] 
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)                    
        last = smoothed_val                                
    return smoothed

def plot_rewards(rewards,title="learning curve"):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{title}")
    plt.xlim(0, len(rewards), 10)  # 设置x轴的范围
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()

def print_cfgs(cfg):
    ''' 打印参数
    '''
    cfg_dict = vars(cfg)
    print("Hyperparameters:")
    print(''.join(['=']*80))
    tplt = "{:^20}\t{:^20}\t{:^20}"
    print(tplt.format("Name", "Value", "Type"))
    for k,v in cfg_dict.items():
        if v.__class__.__name__ == 'list':
            v = str(v)
        print(tplt.format(k,v,str(type(v))))   
    print(''.join(['=']*80))

    # 获取参数
cfg = Config() 
print_cfgs(cfg)
# 训练
env, agent = env_agent_config(cfg)
res_dic = train(cfg, env, agent)
 
plot_rewards(res_dic['rewards'], title=f"training curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}")  
# 测试
res_dic = test(cfg, env, agent)
plot_rewards(res_dic['rewards'], title=f"testing curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}")  # 画出结果