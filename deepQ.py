import gym
import tensorflow as tf
import math
import random
import numpy as np
import keras
import os



REPLAY_MEMORY = 1000000
BATCH_SIZE = 100
 
class deep_learner():

    def __init__(self,observation_space, action_space, weight_file):
        self.observation_space = observation_space
        self.action_space = action_space
        self.weight_file = weight_file


        self.step = 0
        self.epislon = .99
        self.epislon_decay_time = 100
        self.epislon_decay_rate = 0.999
        self.gamma = .99

        self.replayMemory = []

        self.input_width = observation_space.shape[0]
        self.output_width = action_space.n


        #Build the network
        self.model = self.build_net( self.input_width,              #Input Width
                                     self.output_width,             #Output Width
                                     32)                            #Hidden Nodes)

        self.load_weights()


        #Create and ititilize session
        print("Start session and initialize all variables")
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())



        keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=5, write_graph=True, write_images=False)


    def build_net(self, input_width, output_width, hidden_nodes):

        #Add the layers to the network
        model = keras.models.Sequential()
        model.add(keras.layers.core.Dense(hidden_nodes, activation='relu', input_dim=input_width))
        #model.add(keras.layers.core.Dense(hidden_nodes, activation='relu'))
        model.add(keras.layers.core.Dense(output_width, activation='relu'))

        #Build the network with the atom optimizer and mse loss
        model.compile('adam', loss = 'mse')

        self.cbks = [callbacks.ModelCheckpoint(filepath, monitor=monitor,
                save_best_only=save_best_only, mode=mode)]

        return model

    def get_action(self, observation):
        action = None

        #print(self.epislon)
        self.epislon = 0

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


    def train(self):

        batch_size = min(len(self.replayMemory),BATCH_SIZE)

        mini_batch = random.sample(self.replayMemory,batch_size)
        X_train = np.zeros((batch_size, self.input_width))
        Y_train = np.zeros((batch_size, self.output_width))
        loss = 0
        for index_rep in range(batch_size):
          old_rep_state, new_rep_state, action_rep, reward_rep, done_rep  = mini_batch[index_rep]
          old_q = self.model.predict(old_rep_state.reshape(1, self.input_width))[0]
          new_q = self.model.predict(new_rep_state.reshape(1, self.input_width))[0]
          update_target = np.copy(old_q)
          if done_rep:
            update_target[action_rep] = -1
          else:
            update_target[action_rep] = reward_rep + (self.gamma * np.max(new_q))
          X_train[index_rep] = old_rep_state
          Y_train[index_rep] = update_target
        
        loss += self.model.train_on_batch(X_train, Y_train, callbacks=self.cbks)

        return loss

    def variable_summaries(var, name):
      """Attach a lot of summaries to a Tensor."""
      with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('stddev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    learner = deep_learner(env.observation_space, env.action_space, "weights.txt")

    max_runs = 1000000
    max_frames = 500

    for run in xrange(max_runs):
            observation = env.reset()
            reward = 0
            done = False
            best_reward = 0


            for frame in xrange(max_frames):
                env.render()
                action = learner.get_action(observation)
                

                #Calculate the running average
                #running_avg = (running_avg * (frame-1)/frame + reward/frame)


                if done or frame == max_frames-1:
                    print "{0} - {1}".format(run, frame)
                    break

                new_observation, reward, done, info = env.step(action)
                learner.add_memory(observation, new_observation, action, reward, done)

                observation = new_observation
                best_reward += reward

            print("loss", learner.train())

            if(run%100 == 0):
                learner.save_weights()

    print("Saving weights and quitting")
    learner.save_weights()