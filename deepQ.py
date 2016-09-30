import gym
import tensorflow as tf
import math
import random

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
        layers = []

        with tf.name_scope('Input_Vars') as scope:

            inw = tf.Variable(tf.random_uniform([input_width, 128]))
            inb = tf.Variable(tf.random_uniform([128], -1.0, 1.0))

            l1w = tf.Variable(tf.random_uniform([128, 128]))
            l1b = tf.Variable(tf.random_uniform([128], -1.0, 1.0))

            l2w = tf.Variable(tf.random_uniform([128, 128]))
            l2b = tf.Variable(tf.random_uniform([128], -1.0, 1.0))
            
            outw = tf.Variable(tf.random_uniform([128, output_width]))
            outb = tf.Variable(tf.random_uniform([output_width], -1.0, 1.0))

        with tf.name_scope('Input_Vars_2') as scope:

            inwT = tf.Variable(tf.random_uniform([input_width, 128]))
            inbT = tf.Variable(tf.random_uniform([128], -1.0, 1.0))

            l1wT = tf.Variable(tf.random_uniform([128, 128]))
            l1bT = tf.Variable(tf.random_uniform([128], -1.0, 1.0))

            l2wT = tf.Variable(tf.random_uniform([128, 128]))
            l2bT = tf.Variable(tf.random_uniform([128], -1.0, 1.0))
            
            outwT = tf.Variable(tf.random_uniform([128, output_width]))
            outbT = tf.Variable(tf.random_uniform([output_width], -1.0, 1.0))


        self.copyOperation = [inwT.assign(inw),
                                inbT.assign(inb),
                                l1wT.assign(l1w),
                                l1bT.assign(l1b),
                                l2wT.assign(l2w),
                                l2bT.assign(l2b),
                                outwT.assign(outw),
                                outbT.assign(outb)]


        print("Built weights and biases")


        #Define the current Q values
        stateInput = tf.placeholder(tf.float32, [None, input_width])

        layer_in = tf.nn.relu(tf.matmul(stateInput, inw) + inb)
        layer_1 = tf.nn.relu(tf.matmul(layer_in, l1w) + l1b)
        layer_2 = tf.nn.relu(tf.matmul(layer_1, l2w) + l2b)
        QValue = tf.squeeze(tf.matmul(layer_2, outw) + outb)



        #Setup Training And Loss Functions
        actionInput = tf.placeholder(tf.float32, [None, output_width])
        Q_Action = tf.reduce_sum(tf.mul(QValue, actionInput), reduction_indices=1)

        self.yInput = tf.placeholder(tf.float32, [None, ])

        self.loss = tf.reduce_mean(tf.square(Q_Action - self.yInput))
        self.training = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        #Save values for later
        self.stateInput = stateInput
        self.QValue = QValue


    def get_action(self, observation, reward, done):
        action = None

        if(self.step % self.epislon == 0):
            self.epislon *= self.epislon_decay_rate

        #Should we explore?
        if self.epislon > math.random():
            #We should explore
            action = self.action_space.sample()
        else:
            #We sould not explore
            observed_state = np.array([observation])
            action = np.argmax(self.QValue.eval(feed_dict={self.stateInput:[self.currentState]}))

        return action

    def add_memory(self, observation, action, reward, done):
        newState = np.append(observation,self.currentState[:,:,1:],axis = 2)
        self.replayMemory.append((self.currentState,action,reward,newState,done))

        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.pop(0)

        self.step += 1

    def setInitState(self,observation):
        self.currentState = np.stack((observation, observation, observation, observation), axis = 2)


    def train(self):
        if(len(self.replayMemory) < BATCH_SIZE):
            return

        batch = np.random.choice( self.replayMemory, BATCH_SIZE, replace=True)

        commands = [self.loss,self.training]

        feed_dictionary = {currentState: batch[0], action: batch[1], reward: batch[2], newState: batch[3], done: batch[4]}

        self.session.run(commands, feed_dict=feed_dictionary)
		
    def reset():
        self.prev_state

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    learner = deep_learner(env.observation_space, env.action_space)

    max_runs = 100
    max_frames = 50

    for run in xrange(max_runs):
            observation = env.reset()
            reward = 0.0
            done = False

            best_reward = 0

            for frame in xrange(max_frames):
                action = learner.get_action(observation, reward, done)

                #Calculate the running average
                #running_avg = (running_avg * (frame-1)/frame + reward/frame)


                if done or frame == max_frames:
                    print "{0} - {1} - {2}".format(run, frame, best_reward)
                    break

                observation, reward, done, info = env.step(action)
                learner.add_memory(observation, reward, done)
                best_reward+=reward

            learner.train()

                