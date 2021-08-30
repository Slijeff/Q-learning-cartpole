import gym
from collections import deque
import NNmodel
import numpy as np

env = gym.make('CartPole-v1')

EPSILON = 1
MAX_EPSILON = 1
MIN_EPSILON = 0.01
DECAY_RATE = 0.01

EPISODES = 500
TIMESTEPS = 500

# Initialize two network
pred_model = NNmodel.init_nn(env.observation_space.shape, env.action_space.n)
target_model = NNmodel.init_nn(env.observation_space.shape, env.action_space.n)
# synchronize target_model's weight with that of pred_model
target_model.set_weights(pred_model.get_weights())

memory = deque(maxlen=20000)
weight_update_counter = 0

for episode in range(EPISODES):
    sum_rewards = 0
    observations = env.reset()
    done = False
    while not done:

        weight_update_counter += 1
        env.render()

        roll_dice = np.random.rand()
        # Epsilon Greedy Exploration Strategy
        if roll_dice <= EPSILON:
            # Exploration
            action = env.action_space.sample()
        else:
            # Exploitation
            observation_reshaped = observations.reshape([1, observations.shape[0]])
            action = np.argmax(pred_model.predict(observation_reshaped).flatten())
        
        # take action and obtain observations
        new_observations, reward, done, info = env.step(action)
        memory.append([observations, action, reward, new_observations, done])

        # train the network every four timesteps
        if weight_update_counter % 4 == 0 or done:
            NNmodel.train(memory, pred_model, target_model, done)
        
        observations = new_observations
        sum_rewards += reward


        if done:
            print('Total training rewards: {} after n steps = {} with final reward = {}'.format(sum_rewards, episode, reward))
            sum_rewards += 1

            # copy the weights of the prediction model to the target model
            if weight_update_counter >= 100:
                print('Copying main network weights to the target network weights')
                target_model.set_weights(pred_model.get_weights())
                weight_update_counter = 0
            break
    EPSILON = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * episode)
env.close()




