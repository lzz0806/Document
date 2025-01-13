class AlgoConfig:

     def __init__(self, learning_rate=0.01):
          self.learning_rate = learning_rate
          self.epsilon_start = 0
          self.epsilon_end = 0
          self.epsilon_decay = 0  # epsilon衰减系数
          self.batch_size = 64
          self.target_update = 4


class EnvConfig:

     def __init__(self, max_steps=1000):
          # 环境参数
          self.actions_dim = 0
          self.max_steps = 100
          