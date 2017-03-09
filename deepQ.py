import os
import random

import gym
from gym import wrappers
import keras
import numpy as np
import tensorflow as tf


REPLAY_MEMORY = 1000000
BATCH_SIZE = 100
EPISLON_MIN = .1


class deep_learner():

    def __init__(self, observation_space, action_space, weight_file):
        self.observation_space = observation_space
        self.action_space = action_space
        self.weight_file = weight_file
        self.step = 0
        self.epislon = .99
        self.epislon_decay_rate = 0.9995
        self.gamma = .99
        self.replayMemory = []
        self.input_width = observation_space.shape[0]
        self.output_width = action_space.n

        # Build the network
        self.model = self.build_net(self.input_width,   # Input Width
                                    self.output_width,  # Output Width
                                    200)                # Hidden Nodes

        # Create and ititilize session
        print("Start session and initialize all variables")
        self.session = tf.InteractiveSession()
        initializer = tf.global_variables_initializer()
        self.session.run(initializer)

    def build_net(self, input_width, output_width, hidden_nodes):

        # Add the layers to the network
        model = keras.models.Sequential()
        model.add(keras.layers.core.Dense(hidden_nodes, activation='relu',
                  input_dim=input_width))
        model.add(keras.layers.core.Dense(hidden_nodes, activation='relu'))
        model.add(keras.layers.core.Dense(output_width, activation='relu'))

        # Build the network with the atom optimizer and mse loss
        model.compile('adam', loss='mse')
        return model

    def get_action(self, observation):
        action = None

        self.epislon *= self.epislon_decay_rate
        if(self.epislon < EPISLON_MIN):
            self.epislon = EPISLON_MIN

        # Should we explore?
        if self.epislon > random.random():
            # We should explore
            action = self.action_space.sample()
        else:
            # We sould not explore
            action = self.model.predict(
                observation.reshape(1, self.input_width))
            action = np.argmax(action)

        return action

    def add_memory(self, observation, new_observation, action, reward, done):
        self.replayMemory.append((observation, new_observation,
                                  action, reward, done))

        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.pop(0)

        self.step += 1

    def load_weights(self):
        if os.path.isfile(self.weight_file):
            print("Loaded weights from file")
            self.model.load_weights(self.weight_file)

    def save_weights(self):
        print("Saving wights to file")
        if os.path.isfile(self.weight_file):
            os.remove(self.weight_file)
        self.model.save_weights(self.weight_file)

    def get_epislon(self):
        return self.epislon

    def train(self):

        batch_size = min(len(self.replayMemory), BATCH_SIZE)

        mini_batch = random.sample(self.replayMemory, batch_size)
        X_train = np.zeros((batch_size, self.input_width))
        Y_train = np.zeros((batch_size, self.output_width))
        loss = 0
        for index_rep in range(batch_size):
            old_state, new_state, action_rep, \
                reward_rep, done_rep = mini_batch[index_rep]

            update_target = np.copy(self.model.predict(
                old_state.reshape(1, self.input_width))[0])

            if done_rep:
                update_target[action_rep] = reward_rep
            else:
                update_target[action_rep] = reward_rep + (
                    self.gamma * np.max(self.model.predict(
                        new_state.reshape(1, self.input_width))[0]))

            X_train[index_rep] = old_state
            Y_train[index_rep] = update_target

        loss += self.model.train_on_batch(X_train, Y_train)

        return loss


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)
    learner = deep_learner(env.observation_space,
                           env.action_space, "weights.txt")
    max_runs = 1001
    score = [1]

    for run in range(max_runs):
            observation = env.reset()
            reward = 0
            done = False
            best_reward = 0
            running_avg = sum(score)/len(score)
            frame = 0

            while True:
                action = learner.get_action(observation)
                if done:
                    print("{0} - {1} - {2:.2f} - {3:.2f}".format(
                        run, frame, running_avg, learner.get_epislon()))
                    break

                new_observation, reward, done, info = env.step(action)
                learner.add_memory(observation, new_observation,
                                   action, reward, done)

                observation = new_observation
                best_reward += reward

                learner.train()
                frame += 1

            score.append(best_reward)

            if(len(score) > 100):
                score.pop(0)

    print("Saving weights and quitting")
    learner.save_weights()
    env.close()

