# run CartPole with trained DQN model
import torch
import gym
import time
from DQN import normalize_state, state_reward, MLP

# GPU
use_cuda = True
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

# random seed
torch.manual_seed(2023)

# game environment
env = gym.make('CartPole-v1', render_mode="rgb_array")  # animation mode
env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.RecordVideo(env, f"videos/{gym.__version__}")

# load policy net
policy_net = MLP(len(env.observation_space.high), env.action_space.n)
policy_net.load_state_dict(torch.load('policy_net.pth'))
policy_net.to(device)
policy_net.eval()

# game start
state, _ = env.reset(seed=2023)
normalize_state(state)
terminated = False
truncated = False
score = 0
reward = 0
while not (terminated or truncated):
    state = torch.tensor(state, dtype=torch.float32)
    action = torch.argmax(policy_net(state)).item()
    state, env_reward, terminated, truncated, _ = env.step(action)
    normalize_state(state)
    score += env_reward
    reward += state_reward(state, env_reward)

    if truncated:
        print("The game stop due to time limited")

    time.sleep(0.01)

print(f'score: {score} - reward: {reward}')


