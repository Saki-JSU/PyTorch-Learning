# DQN for online CartPole
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import gym
from collections import deque
import random

# GPU
use_cuda = True
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

# random seed
torch.manual_seed(2023)
random.seed(2023)

# Hyper-parameters
episode = 100   # iteration times
gamma = 0.99  # discount rate
learning_rate = 1e-4   # learning rate
memory_len = 10000  # buffer size

# game environment
# env = gym.make('CartPole-v1', render_mode="human")  # animation mode
env = gym.make('CartPole-v1')
# env.observation_space.high outputs the range of each element of states
n_features = len(env.observation_space.high)
n_actions = env.action_space.n

# create empty buffer
# each memory entry is in form: (state, action, env_reward, next_state)
memory = deque(maxlen=memory_len)

# trick 1: normalize function to balance the values of elements in state
def normalize_state(state):
    state[0] /= 2.5
    state[1] /= 2.5
    state[2] /= 0.3
    state[3] /= 0.3

# trick 2: re-define rewards with penalty on position of cart and angle of pole
def state_reward(state, env_reward):
    return env_reward - (abs(state[0]) + abs(state[2])) / 2.5

# epsilon-greedy algorithm
epsilon = 1  # initial epsilon
min_epsilon = 0.1
epsilon_decay = 0.9 / 2.5e3
def get_action(state, epsilon):
    if random.random() < epsilon:
        action = random.randrange(0, n_actions)    # explore
    else:
        state = torch.tensor(state, dtype=torch.float32, device=device)
        action = policy_net(state).argmax().item()

    return action

# transform experience into torch tensor
def get_states_tensor(sample, index):
    sample_len = len(sample)
    states_tensor = torch.empty((sample_len, n_features), dtype=torch.float32, requires_grad=False)

    for i in range(sample_len):
        states_tensor[i, :] = torch.tensor(sample[i][index], dtype=torch.float32)

    return states_tensor

# policy and target Q-networks
# fully-connected MLP
class MLP(nn.Module):
    def __init__(self, input_features, output_values):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=input_features, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=output_values)

    def forward(self, x):
        # trick 3: use selu function, rather than relu or leaky_relu
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = self.fc3(x)
        return x

# train Q network
def fit(model, inputs, labels):
    inputs = inputs.to(device)
    labels = labels.to(device)
    train_ds = TensorDataset(inputs, labels)
    train_dl = DataLoader(train_ds, batch_size=5)

    # train Q network
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    model.train()
    total_loss = 0.0

    for x, y in train_dl:
        out = model(x)
        loss = criterion(out, y)

        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()

    return total_loss / len(inputs)

# optimization
def optimize_model(batch_size=100):
    # sample from buffer
    batch_size = min(batch_size, len(memory))
    train_sample = random.sample(memory, batch_size)

    state = get_states_tensor(train_sample, 0)  # get current state
    next_state = get_states_tensor(train_sample, 3)  # get next state

    q_estimates = policy_net(state.to(device)).detach()
    next_state_q_estimates = target_net(next_state.to(device)).detach()

    for i in range(len(train_sample)):
        # update a better policy pi
        q_estimates[i][train_sample[i][1]] = train_sample[i][2] + gamma * next_state_q_estimates[i].max()

    # train Q network
    fit(policy_net, state, q_estimates)

# initial networks
policy_net = MLP(n_features, n_actions).to(device)
target_net = MLP(n_features, n_actions).to(device)
criterion = nn.MSELoss()

# train one episode
batch_size = 100
def train_one_episode():
    global epsilon
    current_state, _ = env.reset(seed=2023)  # get state
    normalize_state(current_state)   # normalization
    terminated = False
    truncated = False
    score = 0
    reward = 0
    while not (terminated or truncated):
        # one step in the game
        action = get_action(current_state, epsilon)   # choose action with epsilon-greedy algorithm
        next_state, env_reward, terminated, truncated, info = env.step(action)  # attain next state
        normalize_state(next_state)
        new_reward = state_reward(next_state, env_reward)  # new reward with penalty
        memory.append((current_state, action, new_reward, next_state))   # add to buffer
        current_state = next_state
        score += env_reward
        reward += new_reward

        # optimize the model
        optimize_model(batch_size=batch_size)

        # update epsilon
        epsilon -= epsilon_decay

    return score, reward

# test mode
def test():
    state, _ = env.reset(seed=2023)
    normalize_state(state)
    terminated = False
    truncated = False
    score = 0
    reward = 0
    while not (terminated or truncated):
        action = get_action(state, epsilon)
        state, env_reward, terminated, truncated, _ = env.step(action)
        normalize_state(state)
        score += env_reward
        reward += state_reward(state, env_reward)

    return score, reward

# main function
episode_limit = 100
target_update_delay = 2  # update target net every target_update_delay episodes
test_delay = 20  # check test mode every test_delay episodes
def main():
    best_test_reward = 0

    # train mode
    for i in range(episode_limit):
        score, reward = train_one_episode()

        print(f'Episode {i + 1}: score: {score} - reward: {reward}')

        # update target net
        if i % target_update_delay == 0:
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()

        if (i + 1) % test_delay == 0:
            test_score, test_reward = test()
            print(f'Test Episode {i + 1}: test score: {test_score} - test reward: {test_reward}')
            if test_reward > best_test_reward:
                print('New best test reward. Saving model')
                best_test_reward = test_reward
                torch.save(policy_net.state_dict(), 'policy_net.pth')


    # test mode
    test_score, test_reward = test()
    print(f'Test Episode {i + 1}: test score: {test_score} - test reward: {test_reward}')
    if test_reward > best_test_reward:
        print('New best test reward. Saving model')
        best_test_reward = test_reward
        torch.save(policy_net.state_dict(), 'policy_net.pth')

    print(f'best test reward: {best_test_reward}')


if __name__ == '__main__':
    main()







