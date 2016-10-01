import gym
import tensorflow as tf
import math
import random
import numpy as np
import keras as K


REPLAY_MEMORY = 1000000
BATCH_SIZE = 32
 
class deep_learner():

    def __init__(self,observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

        self.step = 0
        self.epislon = .99
        self.epislon_decay_time = 100
        self.epislon_decay_rate = 0.95

        self.replayMemory = []


        #Build the network
        self.build_net( observation_space.shape[0],     #Input Width
                        action_space.n,                 #Output Width
                        2,                              #Hidden Nodes
                        .0001)                          #Learning Rate



        #Create and ititilize session
        print("Start session and initialize all variables")
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

        #create tensorflow graph
        merged_summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter('logs/',self.session.graph)


    def build_net(self,input_width,output_width, num_of_hidden_layers, learning_rate):

        #Create a list to hold the weights and biases
        model = K.models.Sequential()
        model.add(K.layers.core.Dense(16, activation='relu', input_dim=(None,input_width)))
        model.add(K.layers.core.Dense(16, activation='relu'))
        model.add(K.layers.core.Dense(output_width, activation='relu'))

        model.compile('adam', loss = 'mse')

        return model

    def get_action(self, observation):
        action = None

        if(self.step % self.epislon == 0):
            self.epislon *= self.epislon_decay_rate

        #Should we explore?
        if self.epislon > random.random():
            #We should explore
            action = self.action_space.sample()
        else:
            #We sould not explore
            l = []
            l.extend(observation)
            #observed_state = np.array([observation])
            action = self.QValue.eval(feed_dict={self.stateInput:[l]})
            action = np.argmax(action)

        return action

    def add_memory(self,observation, new_observation, action, reward):
        self.replayMemory.append((observation, action, reward, new_observation))

        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.pop(0)

        self.step += 1


    def train(self):
        print("Train!")
        if(len(self.replayMemory) < BATCH_SIZE):
            print("Too Small!")
            return

        minibatch = random.sample(self.replayMemory,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        y_batch = []
        QValue_batch = self.inwT.eval(feed_dict={self.stateInputT:nextState_batch})
        for i in range(0,BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))


        self.trainStep.run(feed_dict={
            self.yInput : y_batch,
            self.actionInput : action_batch,
            self.stateInput : state_batch
            })

        self.training.run(feed_dict=batch)
		
    def reset():
        self.prev_state

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    learner = deep_learner(env.observation_space, env.action_space)

    max_runs = 1000
    max_frames = 5000

    for run in xrange(max_runs):
            observation = env.reset()
            print(observation)
            reward = 0.0
            done = False

            best_reward = 0

            for frame in xrange(max_frames):
                env.render()
                action = learner.get_action(observation)

                #Calculate the running average
                #running_avg = (running_avg * (frame-1)/frame + reward/frame)


                if done or frame == max_frames:
                    print "{0} - {1} - {2}".format(run, frame, best_reward)
                    break

                new_observation, reward, done, info = env.step(action)
                learner.add_memory(observation, new_observation, action, reward)
                best_reward+=reward

            learner.train()

                