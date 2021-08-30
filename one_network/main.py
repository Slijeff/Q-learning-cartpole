import gym
from agent import Agent

env = gym.make('CartPole-v1')

EPISODES = 500
TIMESTEPS = 1000

# initialize prediction network and target network
model = Agent(env) 

total_timesteps = 0
for i_episode in range(EPISODES):
    observation = env.reset()
    for t in range(TIMESTEPS):
        total_timesteps += t
        env.render()
        action = model.perform_action(observation) # perform the best action by looking at current observation
        last_observation = observation
        observation, reward, done, info = env.step(action)
        model.memorize(done, action, observation, last_observation)
        model.replay(20)
        # decrease epsilon over time
        # Model.epsilon = Model.epsilon if Model.epsilon < 0.01 else Model.epsilon * 0.99
        if done:
            print('Episode {} ended after {} timesteps, current exploration is {}'.format(i_episode, t+1, model.epsilon))
            break
env.close()