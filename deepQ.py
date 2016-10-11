from __future__ import division
import gym
import tensorflow as tf
import math
import random
import numpy as np
import keras
import os
import pprint
pp = pprint.PrettyPrinter(indent=4)

REPLAY_MEMORY = 1000000
BATCH_SIZE = 1000
EPISLON_MIN = .05
NUM_OF_NODES = 200
TRAINS_UNTILL_COPY = 1000
START_EPISLON = .99
EPISLON_DECAY = .99995
GAMMA = .99

MAX_RUNS = 30000
MAX_FRAMES = 199

PRINT_TRAIN = 1000
PRINT_RENDER = 1000



class deep_learner():

    def __init__(self, observation_space, action_space, weight_file):
        self.observation_space = observation_space
        self.action_space = action_space
        self.weight_file = weight_file
        self.step = 0
        self.epislon = START_EPISLON

        self.replayMemory = []
        self.input_width = observation_space.shape[0]
        self.output_width = action_space.n

        # Build the network
        self.model = self.build_net(self.input_width,   # Input Width
                                    self.output_width,  # Output Width
                                    NUM_OF_NODES)       # Hidden Nodes

        self.model_prime = self.build_net(self.input_width,   # Input Width
                                    self.output_width,        # Output Width
                                    NUM_OF_NODES)             # Hidden Nodes


        self.model_prime.set_weights(self.model.get_weights())

        self.current_weights = self.model.get_weights()
        self.target_weights = self.model.get_weights()

        # Create and ititilize session
        print("Start session and initialize all variables")
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

    def update_weights(self):
        self.model



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

        self.epislon *= EPISLON_DECAY
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
        self.replayMemory.append((observation, new_observation, action, reward, done))

        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.pop(0)


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
        

        if(len(self.replayMemory) < 10):
            return

        #Calculate the batch size from min(current array size or Batch_Size)
        batch_size = min(len(self.replayMemory), BATCH_SIZE)

        #Get a random sample from memory.
        mini_batch = np.array(random.sample(self.replayMemory, batch_size))


        input_batch = np.vstack(mini_batch[:,0])
        reward_batch = np.vstack(mini_batch[:,1])
        #print("Input")
        #pp.pprint(input_batch)


        output_train = np.zeros((batch_size, self.output_width))

        this_reward = self.model_prime.predict(reward_batch, batch_size=batch_size)
        #if(self.step % PRINT_TRAIN == 0):
        #    print("This Reward")
        #    pp.pprint(this_reward)

        
        new_reward = reward + np.multiply(GAMMA, this_reward)
        #if(self.step % PRINT_TRAIN == 0):
        #    print("New Reward")
        #    pp.pprint(new_reward)

        indexes = np.argmax(new_reward, axis=1)
        #if(self.step % PRINT_TRAIN == 0):
        #    print("indexes")
        #    pp.pprint(indexes)

        for index in xrange(len(this_reward)):
            max_action = indexes[index]
            output_train[index][max_action] = new_reward[index][max_action]           

        if(self.step % PRINT_TRAIN == 0):
            print("Output Train")
            pp.pprint(output_train)

        self.model.train_on_batch(input_batch, output_train)

        if(self.step % TRAINS_UNTILL_COPY == 0):
            #self.copy_weights(self.model, self.model_prime)
            self.model_prime.set_weights(self.model.get_weights())

        self.step += 1
 

    def copy_weights(self, model_source, model_target):
        model_source.set_weights(model_target.get_weights())
        print("Updated Weights!")

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.monitor.start('/tmp/cartpole-experiment-1', force=True)
    learner = deep_learner(env.observation_space,
                           env.action_space, "weights.txt")

    score = [1]

    for run in xrange(MAX_RUNS):
            observation = env.reset()
            reward = 0
            done = False
            best_reward = 0
            running_avg = sum(score)/len(score)


            for frame in xrange(MAX_FRAMES):
                if(frame == MAX_FRAMES-1):
                    print("Finished Run!")

                if(run % PRINT_RENDER == 0):
                    env.render()

                action = learner.get_action(observation)
                if done:
                    print "{0} - {1} - {2:.2f} - {3:.2f}".format(
                        run, frame, running_avg, learner.get_epislon())
                    break

                new_observation, reward, done, info = env.step(action)
                learner.add_memory(observation, new_observation,
                                   action, reward, done)

                observation = new_observation
                best_reward += reward

                learner.train()
            score.append(best_reward)

            if(len(score) > 100):
                score.pop(0)

    print("Saving weights and quitting")
    learner.save_weights()

    env.monitor.close()
