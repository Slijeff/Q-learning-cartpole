import torch
import gym
import numpy as np

model = torch.load('atari.pt')

# gym.envs.register(
#         id='CartPole-v2',
#         entry_point='gym.envs.classic_control:CartPoleEnv',
#         max_episode_steps=5000,
#         reward_threshold=475.0,
#     )
# env = gym.make('CartPole-v2')
env = gym.make('BipedalWalker-v3')
state = env.reset()
done = False
action = 0
while not done:
    env.render()
    with torch.no_grad():
        action = np.argmax(model(torch.tensor(state).float()))
    state, reward, done, info = env.step(action.item())