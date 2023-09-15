import random
import torch

class Memory_Buffer(object):
    def __init__(self, memory_size=100000):
        self.buffer = []
        self.memory_size = memory_size
        self.next_idx = 0

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.buffer) <= self.memory_size: # buffer not full
            self.buffer.append(data)
        else: # buffer is full
            self.buffer[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):
            idx = random.randint(0, self.size() - 1)
            data = self.buffer[idx]
            state, action, reward, next_state, done = data
            states.append(torch.from_numpy(state.__array__()[None]/255).float())
            actions.append(action)
            rewards.append(reward)
            next_states.append(torch.from_numpy(next_state.__array__()[None]/255).float())
            dones.append(done)

        return torch.cat(states), actions, rewards, torch.cat(next_states), dones

    def size(self):
        return len(self.buffer)
