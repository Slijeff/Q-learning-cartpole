import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

def init_nn(in_shape, out_shape):
    
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=in_shape, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(24, input_shape=in_shape, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(out_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model

def train(memory, prediction_model, target_model, done):
    learning_rate = 0.7
    gamma = 0.7
    
    replay_size = 1000
    if len(memory) < replay_size:
        return
    
    sample_size = 128
    sample = random.sample(memory, sample_size)

    current_observation = np.array([memo[0] for memo in sample])
    current_q_scores = prediction_model.predict(current_observation)

    new_observations = np.array([memo[3] for memo in sample])
    target_q_scores = target_model.predict(new_observations)

    X = []
    Y = []
    for i, (observation, action, reward, new_observation, done) in enumerate(sample):
        if not done:
            
            # Bellman equation
            max_future_q = reward + gamma * np.max(target_q_scores[i])
        else:
            max_future_q = reward
        
        current_q_score = current_q_scores[i]
        # replace output in the prediction network regarding the learning rate and perform back propagation to retrain
        current_q_score[action] = (1 - learning_rate) * current_q_score[action] + learning_rate * max_future_q

        X.append(observation)
        Y.append(current_q_score)

    prediction_model.fit(np.array(X), np.array(Y), batch_size=sample_size, verbose=0, shuffle=True)