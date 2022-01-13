import torch.nn as nn
import torch.nn.functional as functional
import gym
import random
import torch
import numpy as np
import pygad.torchga as ga
from nn import Network

INPUT_DIM = 4
OUTPUT_DIM = 2
EPOCH = 2000
POP_SIZE = 20
MUTATION_RATE = 0.2
GOAL_REWARD = 100
MODEL_NAME = 'model.pt'

def init_population(pop_size, input_dim, output_dim):
    population = []
    for _ in range(pop_size):
        population.append(Network(input_dim, output_dim).apply(init_weights))
    return population

def init_weights(model):
    if isinstance(model, nn.Linear):
        nn.init.uniform_(model.weight)

def evaluate(model, env, n_games=50, render=False):
    total_reward = 0
    for i in range(n_games):
        state = env.reset()
        done = False
        while not done:
            if render and i % n_games == 0:
                env.render()
            action = 0
            with torch.no_grad():
                action = np.argmax(model(torch.tensor(state).float()))
                action = action.item()
            state, reward, done, _ = env.step(action)
            total_reward += reward
    return total_reward / n_games

def crossover(model1, model2):
    weights1 = ga.model_weights_as_vector(model1)
    weights2 = ga.model_weights_as_vector(model2)
    random_idx = random.randint(0, len(weights1))
    return np.append(weights1[:random_idx], weights2[random_idx:])

def mutate(weights):
    new_weights = [0 for _ in range(len(weights))]
    for i in range(len(weights)):
        if random.random() <= MUTATION_RATE:
            new_weights[i] = weights[i] + np.random.rand()
    return new_weights

def next_generation(curr_gen, env, render=False):
    new_gen = []
    rewards = []
    for individual in curr_gen:
        rewards.append(evaluate(individual, env, 100, render=render))
        
    first, second = np.argsort(rewards)[-2:]
    print("Best:", rewards[first], "Second:", rewards[second])
    
    # elitism
    new_gen.append(curr_gen[first])
    new_gen.append(curr_gen[second])

    for _ in range(len(curr_gen) - 2):
        crossed = crossover(curr_gen[first], curr_gen[second])
        mutated = mutate(crossed)
        new_mdl = Network(INPUT_DIM, OUTPUT_DIM)
        new_weights = ga.model_weights_as_dict(new_mdl, mutated)
        new_mdl.load_state_dict(new_weights)
        new_gen.append(new_mdl)
    
    return rewards[first], rewards[second], sum(rewards) / POP_SIZE, new_gen

def best_model(population, env):
    rewards = []
    for individual in population:
        rewards.append(evaluate(individual, env, 100, render=False))
    return np.argsort(rewards)[-1]

if __name__ == "__main__":
    global avg_best_reward
    avg_best_reward = 0
    POPULATION = init_population(POP_SIZE, INPUT_DIM, OUTPUT_DIM)
    
    gym.envs.register(
        id='CartPole-v2',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=500,
        reward_threshold=475.0,
    )
    env = gym.make('CartPole-v2')
    # env = gym.make('MountainCar-v0')

    sum_reward = 0
    f,s = 0,0
    for i in range(EPOCH):
        
        f,s,sum_reward, POPULATION = next_generation(POPULATION, env)

        if f + s >= GOAL_REWARD * 2:
            print("Solved in %d steps" % i)
            # save model
            torch.save(POPULATION[best_model(POPULATION, env)], MODEL_NAME)
            break
        print(f"Generation {i}: Average reward recieved: {sum_reward}")
    