import os.path

from RL.algo.PPO import PPO, PPOAgent
from RL.env import Env
from RL.config import AlgoConfig, EnvConfig
from RL.utils import plot_rewards
import warnings

train = True
warnings.filterwarnings("ignore")

algo_config = AlgoConfig()
env_config = EnvConfig()
env_config.env_name = 'CartPole-v1'
env = Env(env_config)
env = env.make_env()
ppo = PPO(algo_config, env_config)
ppo_agent = PPOAgent(algo_config, env_config)
if train:
    agent, info = ppo.train(env, ppo_agent)
    ppo_agent.save_model("")
    plot_rewards(info["rewards"])
else:
    model_path = "ppo_agent_model.pth"
    ppo_agent.load_model(model_path)
    ppo.evaluate(env, ppo_agent)
