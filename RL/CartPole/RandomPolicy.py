# random policy
import gym

# initialize game environment
env = gym.make("CartPole-v1", render_mode="human")

# restart game
env.reset()
# show the game image
env.render()
# update
for _ in range(100):
    # sample a random action
    action = env.action_space.sample()
    # interact with environment, and outputs state, rewards
    # terminated: died in the game
    # truncated: out of epoches
    states, reward, terminated, truncated, info = env.step(action)

    # if died, reset the game
    if terminated or truncated:
        env.reset()
        env.render()

# close game
env.close()

