# Atari Pong game

import numpy as np
import math
import matplotlib.pyplot as plt

import torch.autograd as autograd

import gym
from gym.wrappers import AtariPreprocessing, FrameStack

from models import *
from memory import *

# epsilon-greedy algorithm
def select_action(state, epsilon_max=1, epsilon_min=0.05, eps_decay=30000):
    global steps_done
    # e-greedy decay
    epsilon = epsilon_min + (epsilon_max - epsilon_min) * math.exp(
        -1. * steps_done / eps_decay)
    steps_done += 1

    if random.random() < epsilon:
        action = random.randrange(6)
    else:
        action = policy_net(state).argmax()
    return action

# optimization
def optimize_model(batch_size):
    if memory_buffer.size() > batch_size:
        states, actions, rewards, next_states, dones = memory_buffer.sample(batch_size)
        # get tensor
        states = states.to(device)
        actions = torch.tensor(actions).long().to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states = next_states.to(device)
        is_done = torch.tensor(dones).bool().to(device)

        # get q-values for all actions in current states
        predicted_qvalues = policy_net(states)

        # select q-values for chosen actions
        predicted_qvalues_for_actions = predicted_qvalues[range(states.shape[0]), actions]

        # compute q-values for all actions in next states
        predicted_next_qvalues = target_net(next_states)

        # compute V*(next_states) using predicted next q-values
        next_state_values = predicted_next_qvalues.max(-1)[0]

        # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
        target_qvalues_for_actions = rewards + gamma * next_state_values

        # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
        target_qvalues_for_actions = torch.where(is_done, rewards, target_qvalues_for_actions)

        # loss function
        loss = F.smooth_l1_loss(predicted_qvalues_for_actions, target_qvalues_for_actions.detach()).to(device)

        # optimizer
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        return loss.item()
    else:
        return 0


# train mode
def train(env, n_steps):
    # game start
    print_interval = 1000
    episode_reward = 0
    episode_num = 0
    states, _ = env.reset(seed=SEED)

    for i in range(n_steps):
        # choose action
        action = select_action(torch.from_numpy(states.__array__()[None] / 255).float().to(device))
        # step
        next_states, reward, terminated, truncated, _ = env.step(action)
        # cumulative reward
        episode_reward += reward
        # add to memory
        memory_buffer.push(states, action, reward, next_states, terminated)
        # update
        states = next_states

        loss = 0
        if memory_buffer.size() >= LEARNING_START:
            loss = optimize_model(batch_size)
            losses.append(loss)

        if i % print_interval == 0:
            print("frames: %5d, reward: %5f, loss: %4f, episode: %4d" % (
                i, np.mean(all_rewards[-10:]), loss, episode_num))

        if i % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if terminated or truncated:
            states, _ = env.reset(seed=SEED)

            all_rewards.append(episode_reward)
            episode_reward = 0
            episode_num += 1


# test mode
def test():
    env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")
    env = AtariPreprocessing(env, scale_obs=False, terminal_on_life_loss=True)
    env = FrameStack(env, num_stack=4)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(env, f"videos/{gym.__version__}", name_prefix="pong.mp4")

    states, _ = env.reset(seed=SEED)
    total_reward = 0.0
    terminated = False
    truncated= False
    while not (terminated or truncated):
        # choose action
        action = select_action(torch.from_numpy(states.__array__()[None] / 255).float().to(device))
        # step
        next_states, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if not (terminated or truncated):
            states = next_states
        else:
            print("Reward: {}".format(total_reward))

    env.close()
    return


# main file
if __name__ == '__main__':
    # random seed
    SEED = 2023
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # GPU setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if device == "cuda" else autograd.Variable(*args, **kwargs)

    # Training DQN in PongNoFrameskip-v4
    env = gym.make('PongNoFrameskip-v4')
    # modify the environment
    env = AtariPreprocessing(env, scale_obs=False, terminal_on_life_loss=True)
    env = FrameStack(env, num_stack=4)

    # hyper-parameter
    gamma = 0.99  # discount rate
    batch_size = 32
    learning_rate = 2e-4
    MEMORY_SIZE = 100000
    # train model
    LEARNING_START = 10000
    TARGET_UPDATE = 1000
    # recorder
    losses = []
    all_rewards = []

    # initialize environment
    policy_net = CNN().to(device)
    target_net = CNN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    memory_buffer = Memory_Buffer(MEMORY_SIZE)
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=learning_rate, eps=0.001, alpha=0.95)

    # train mode
    n_steps = 2000000
    steps_done = 0
    train(env, n_steps)
    torch.save(policy_net, "dqn_pong_model.pth")  # save training results

    # plot
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (n_steps, np.mean(all_rewards[-10:])))
    plt.plot(all_rewards)
    plt.xlabel("episodes")
    plt.subplot(132)
    plt.title('loss')
    plt.xlabel("frames")
    plt.plot(losses)
    plt.savefig('reward and loss.png')
    # plt.show()

    # test mode
    test()


