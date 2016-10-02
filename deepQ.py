from __future__ import division
import gym
import tensorflow as tf
import math
import random
import numpy as np
import keras
import os


REPLAY_MEMORY = 100000
BATCH_SIZE = 10
 
class deep_learner():

    def __init__(self,observation_space, action_space, weight_file):
        self.observation_space = observation_space
        self.action_space = action_space
        self.weight_file = weight_file


        self.step = 0
        self.epislon = .99
        self.epislon_decay_rate = 0.99
        self.gamma = .99

        self.replayMemory = []

        self.input_width = observation_space.shape[0]
        self.output_width = action_space.n


        #Build the network
        self.model = self.build_net( self.input_width,              #Input Width
                                     self.output_width,             #Output Width
                                     200)                           #Hidden Nodes)

        #self.load_weights()


        #Create and ititilize session
        print("Start session and initialize all variables")
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())


    def build_net(self, input_width, output_width, hidden_nodes):

        #Add the layers to the network
        model = keras.models.Sequential()
        model.add(keras.layers.core.Dense(hidden_nodes, activation='relu', input_dim=input_width))
        model.add(keras.layers.core.Dense(hidden_nodes, activation='relu'))
        #model.add(keras.layers.core.Dense(hidden_nodes, activation='relu'))
        model.add(keras.layers.core.Dense(output_width, activation='relu'))

        #Build the network with the atom optimizer and mse loss
        model.compile('adam', loss = 'mse')

        return model

    def get_action(self, observation):
        action = None

        #print(self.epislon)
        self.epislon *= self.epislon_decay_rate

        #Should we explore?
        if self.epislon > random.random():
            #We should explore
            action = self.action_space.sample()
        else:
            #We sould not explore

            action = self.model.predict(observation.reshape(1, self.input_width))
            action = np.argmax(action)

        return action

    def add_memory(self,observation, new_observation, action, reward, done):
        self.replayMemory.append((observation, new_observation, action, reward, done))

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


    def updateTargetQ():
        pass

    def train(self):

        batch_size = min(len(self.replayMemory),BATCH_SIZE)

        mini_batch = random.sample(self.replayMemory, batch_size)
        X_train = np.zeros((batch_size, self.input_width))
        Y_train = np.zeros((batch_size, self.output_width))
        #print(X_train)
        #print(Y_train)
        loss = 0
        for index_rep in range(batch_size):
          old_state, new_state, action_rep, reward_rep, done_rep  = mini_batch[index_rep]
          #print(old_state, new_state, action_rep, reward_rep, done_rep)

          update_target = np.copy(self.model.predict(old_state.reshape(1, self.input_width))[0])
          #print("update_target",update_target)

          if done_rep:
            update_target[action_rep] = reward_rep
          else:
            update_target[action_rep] = reward_rep + (self.gamma 
                * np.max(self.model.predict(new_state.reshape(1, self.input_width))[0]))

          #print('update_target2',update_target)
          X_train[index_rep] = old_state
          Y_train[index_rep] = update_target

        
        loss += self.model.train_on_batch(X_train, Y_train)

        return loss


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    env.monitor.start('/tmp/cartpole-experiment-1', force=True)

    learner = deep_learner(env.observation_space, env.action_space, "weights.txt")

    max_runs = 1000
    max_frames = 500

    score = [1]

    for run in xrange(max_runs):
            observation = env.reset()
            reward = 0
            done = False
            best_reward = 0
            running_avg = sum(score)/len(score)


            for frame in xrange(max_frames):
                if(frame == max_frames-1):
                    print("Finished Run!")
                #env.render()
                action = learner.get_action(observation)
                

                #Calculate the running average
                

                if done:
                    print "{0} - {1} - {2}".format(run, frame, running_avg)
                    break

                new_observation, reward, done, info = env.step(action)
                learner.add_memory(observation, new_observation, action, reward, done)

                observation = new_observation
                best_reward += reward
                


                learner.train()


            
            score.append(best_reward)


            if(len(score)>100):
                score.pop(0)

            #if(run%100 == 0):
            #    learner.save_weights()

    print("Saving weights and quitting")
    learner.save_weights()

    env.monitor.close()