from torch.distributions import Categorical
from RL.config import AlgoConfig, EnvConfig
from RL.network.ppo_network import Actor, Critic
from RL.replay_buffer import PGReplay
import copy
import torch

class PPOAgent:

    def __init__(self, algo_config: AlgoConfig, env_config: EnvConfig):
        self.memory = PGReplay()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(env_config.n_states, env_config.n_actions).to(self.device)
        self.critic = Critic(env_config.n_states, 1).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=algo_config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=algo_config.learning_rate)
        self.k_epochs = algo_config.k_epochs
        self.eps_clip = algo_config.eps_clip  # clip parameter for PPO
        self.entropy_coef = algo_config.entropy_coef  # entropy coefficient
        self.sample_count = 0
        self.update_freq = algo_config.update_freq
        self.log_probs = None
        self.gamma = algo_config.gamma


    def save_model(self, path):
        torch.save({
            "actor_model": self.actor.state_dict(),
            "critic_model": self.critic.state_dict(),
        },
            path + "ppo_agent_model.pth")

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor_model"])
        self.critic.load_state_dict(checkpoint["critic_model"])

    def sample_action(self, state):
        self.sample_count += 1
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        prob = self.actor(state)
        dist = Categorical(prob)
        action = dist.sample()
        self.log_probs = dist.log_prob(action).detach()
        return action.detach().cpu().numpy().item()

    @torch.no_grad()
    def predict_action(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        prob = self.actor(state)
        dist = Categorical(prob)
        action = dist.sample()
        return action.detach().cpu().numpy().item()

    def update(self):
        # # update policy every n steps
        if self.sample_count % self.update_freq != 0:
            return
        old_state, old_action, old_log_probs, old_rewards, old_done = self.memory.sample()
        old_state = torch.tensor(old_state, device=self.device, dtype=torch.float32)
        old_action = torch.tensor(old_action, device=self.device, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, device=self.device, dtype=torch.float32)
        rewards = []
        discount_sum = 0
        for reward, done in zip(reversed(old_rewards), reversed(old_done)):
            if done:
                discount_sum = 0
            discount_sum = reward + self.gamma * discount_sum
            rewards.insert(0, discount_sum)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        for _ in range(self.k_epochs):
            # compute advantage
            values = self.critic(old_state)  # detach to avoid backprop through the critic
            advantage = rewards - values.detach()
            # get action probabilities
            probs = self.actor(old_state)
            dist = Categorical(probs)
            # get new action probabilities
            new_probs = dist.log_prob(old_action)
            # compute ratio (pi_theta / pi_theta__old):
            ratio = torch.exp(new_probs - old_log_probs)  # old_log_probs must be detached
            # compute surrogate loss
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            # compute actor loss
            actor_loss = -torch.min(surr1, surr2).mean() + self.entropy_coef * dist.entropy().mean()
            # compute critic loss
            critic_loss = (rewards - values).pow(2).mean()
            # take gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        self.memory.clear()

class PPO:

    def __init__(self, ppo_algo_config: AlgoConfig, ppo_env_config: EnvConfig):
        self.ppo_algo_config = ppo_algo_config
        self.ppo_env_config = ppo_env_config

    def train(self, env, agent: PPOAgent):
        print("PPO Trainer start Training...")
        if not isinstance(agent, PPOAgent):
            raise TypeError(f"This agent must be of type PPOAgent, bug get {agent.__class__.__name__}")
        rewards = []
        steps = []
        best_episode_reward = 0
        output_agent = None
        for episode in range(self.ppo_env_config.train_episode):
            # 训练的最大局数
            episode_reward = 0  # 记录每一局的reward
            episode_step = 0    # 记录每一局的step
            state = env.reset()
            for _ in range(self.ppo_env_config.max_steps):
                # env.render()
                # 每一局游戏最大的step
                episode_step += 1
                # 将state输入到agent的网络，并用sample方法选择一个action
                action = agent.sample_action(state)
                next_state, reward, done, _ = env.step(action)
                # 将当前的state、action、action的对数几率、奖励、done输入memory
                agent.memory.push((state, action, agent.log_probs, reward, done))
                state = next_state
                agent.update()
                episode_reward += reward  # 累加每一步的奖励
                if done:
                    # 如果环境done 结束这一局
                    break
            if (episode + 1) % self.ppo_env_config.eval_per_episode == 0:
                # 局数满足设定的评估条件后，对agent进行评估
                sum_eval_episode_reward = 0
                for _ in range(self.ppo_env_config.eval_episode):
                    # 评估的回合数
                    eval_episode_reward = 0
                    state = env.reset()
                    for _ in range(self.ppo_env_config.max_steps):
                        # 评估，不做反向传播
                        action = agent.predict_action(state)
                        next_state, reward, done, _ = env.step(action)
                        state = next_state
                        eval_episode_reward += reward
                        if done:
                            break
                    sum_eval_episode_reward += eval_episode_reward
                # 计算奖励均值
                mean_eval_episode_reward = sum_eval_episode_reward / self.ppo_env_config.eval_episode
                if mean_eval_episode_reward >= best_episode_reward:
                    best_episode_reward = mean_eval_episode_reward
                    output_agent = copy.deepcopy(agent)
                    print(f"回合：{episode+1}/{self.ppo_env_config.max_steps}，奖励：{episode_reward:.2f}，评估奖励：{mean_eval_episode_reward:.2f}，最佳评估奖励：{mean_eval_episode_reward:.2f}，更新模型！")
                else:
                    print(f"回合：{episode+1}/{self.ppo_env_config.max_steps}，奖励：{episode_reward:.2f}，评估奖励：{mean_eval_episode_reward:.2f}，最佳评估奖励：{mean_eval_episode_reward:.2f}")
            steps.append(episode_step)
            rewards.append(episode_reward)
        print("PPO Trainer end Training...")
        env.close()
        return output_agent, {"rewards": rewards, "steps":steps}

    def evaluate(self, env, agent: PPOAgent):
        print("开始测试！")
        rewards = []  # 记录所有回合的奖励
        steps = []
        for i_ep in range(self.ppo_env_config.test_episode):
            ep_reward = 0  # 记录一回合内的奖励
            ep_step = 0
            state = env.reset()  # 重置环境，返回初始状态
            for _ in range(self.ppo_env_config.max_steps):
                env.render()
                ep_step += 1
                action = agent.predict_action(state)  # 选择动作
                next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
                state = next_state  # 更新下一个状态
                ep_reward += reward  # 累加奖励
                if done:
                    break
            steps.append(ep_step)
            rewards.append(ep_reward)
            print(f"回合：{i_ep + 1}/{self.ppo_env_config.test_episode}，奖励：{ep_reward:.2f}")
        print("完成测试")
        env.close()
        return {'rewards': rewards}


if __name__ == '__main__':
    pass