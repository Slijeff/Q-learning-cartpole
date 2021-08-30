from collections import deque
from neural_net import SingleLayer
from activation_functions import relu
import numpy as np

class Agent:

    def __init__(self, environment):
        self.env = environment
        self.epsilon = 1
        self.neuron_size = 48
        self.input_size = environment.observation_space.shape[0]
        self.output_size = environment.action_space.n
        self.hidden_layer_size = 2
        self.layers = [SingleLayer(self.input_size + 1, self.neuron_size, activation = relu)] # setup the input layer
        for _ in range(self.hidden_layer_size - 1):
            self.layers.append(SingleLayer(self.neuron_size + 1, self.neuron_size, activation = relu)) # setup hidden layers
        self.layers.append(SingleLayer(self.neuron_size + 1, self.output_size)) # setup output layer
        self.gamma = 0.95
        self.memory = deque([], 10000)
    
    def perform_action(self, observation):
        vals = self.forward_propagation(observation)
        # exploration vs. exploitation
        if (np.random.random() > self.epsilon):
            return np.argmax(vals)
        else:
            return np.random.randint(self.env.action_space.n)
        
    # propagate forward and reach the output within nn
    def forward_propagation(self, observation, memorize = False):
        vals = np.copy(observation)
        layer_number = 0
        for layer in self.layers:
            vals = layer.forward_propagation_single(vals, memorize) # keep propagating
            layer_number += 1
        return vals

    def backward_propagation(self, calculated_vals, experimental_vals):
        delta = calculated_vals - experimental_vals
        for layer in reversed(self.layers):
            delta = layer.backward_propagation_single(delta)

    def memorize(self, done, action, observation, last_observation):
        self.memory.append([done, action, observation, last_observation])

    def replay(self, updates = 20):
        # check to see if we have enough experiences to start learning
        if (len(self.memory) < updates):
            return
        else:
            batch_indices = np.random.choice(len(self.memory), updates)
            for i in batch_indices:
                done, sel_action, new_observation, prev_observation = self.memory[i]
                # 
                action_vals = self.forward_propagation(prev_observation, memorize=True)
                next_action_vals = self.forward_propagation(new_observation, memorize=False)
                experiment_vals = np.copy(action_vals)
                if done:
                    experiment_vals[sel_action] = -1
                else:
                    experiment_vals[sel_action] = 1 + self.gamma*np.max(next_action_vals)
                
                # The difference between experimental_vals and action_vals will form the basis for updating the weights of our network
                self.backward_propagation(action_vals, experiment_vals)
                self.epsilon = self.epsilon if self.epsilon < 0.01 else self.epsilon * 0.999
            for layer in self.layers:
                layer.learning_rate = layer.learning_rate if layer.learning_rate < 0.0001 else layer.learning_rate * 0.99
    

