from collections import deque
import random

class Buffer:

     def __init__(self, capacity: int):
          self.capacity = capacity
          self.buffer = deque(maxlen=capacity)
     
     def push(self, transaction):
          raise NotImplementedError()

     def sample(self):
          raise NotImplementedError()


class ReplayBuffer(Buffer):

     def __init__(self, capacity: int):
          super().__init__(capacity)
          self.capacity = capacity
          self.buffer = deque(maxlen=capacity)
     
     def push(self, transaction):
          self.buffer.append(transaction)
     
     def sample(self, batch_size: int, sequential: bool=False):
          if batch_size > len(self.buffer):
               batch_size = len(self.buffer)
          if sequential:
               indices = random.sample(self.buffer, batch_size)
               return [self.buffer[i] for i in indices]
          else:
               batch = random.sample(self.buffer, batch_size)
               return zip(*batch)
     
     def clear(self):
          self.buffer.clear()

     def __len__(self):
          return len(self.buffer)
