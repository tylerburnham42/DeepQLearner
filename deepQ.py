import gym
import tensorflow as tf
 
class deep_learner():

    def __init__(self,observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space


        #Build the network
        self.build_net( observation_space.shape[0],     #Input Width
                        action_space.n,                 #Output Width
                        2,                              #Hidden Nodes
                        .0001,                          #Learning Rate
                        .99)                            #Gamma


        #Create and ititilize session
        print("Start session and initialize all variables")
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

        #create tensorflow graph
        merged_summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter('logs/',self.session.graph)


    def build_net(self,input_width,output_width, num_of_hidden_layers, learning_rate, gamma):

        #Create a list to hold the weights and biases
        layers = []

        with tf.name_scope('Input_Vars') as scope:
            #Create input weights
            input_weight_dist = tf.random_uniform([input_width, 128], name="Input_Weight_Random_Dist")
            input_weight_var = tf.Variable(input_weight_dist, name="Input_Weight")

            #Create input biases
            input_bias_dist = tf.random_uniform([128], -1.0, 1.0, name="Input_Bias_Random_Dist")
            input_bias_var = tf.Variable(input_bias_dist, name="Input_Bias")

            #Add the input layer
            layers.append((input_weight_var,input_bias_var))


        #Add <num_of_hidden_layers> extra layers
        for layer_number in xrange(num_of_hidden_layers):
            layer_name = "Layer_{0}_".format(layer_number)

            with tf.name_scope(layer_name + 'Vars') as scope:
                #Create Weights
                weight_dist = tf.random_uniform([128, 128], name=layer_name+"Weight_Random_Dist")
                weight_var = tf.Variable(weight_dist, name=layer_name+"Weight")

                #Create Biases
                bias_dist = tf.random_uniform([128], -1.0, 1.0, name=layer_name+"Bias_Random_Dist")
                bias_var = tf.Variable(bias_dist, name=layer_name+"Bias")

                #Add them to the list as a tuple
                layers.append((weight_var, bias_var))


        with tf.name_scope('Output_Vars') as scope:
            #Output Weights
            output_weight_dist = tf.random_uniform([128, output_width], name="Output_Weight_Random_Dist")
            output_weight_var = tf.Variable(output_weight_dist, name="Output_Weight")

            #Output Biases
            output_bias_dist = tf.random_uniform([output_width], -1.0, 1.0, name="Output_Bias_Random_Dist")
            output_bias_var = tf.Variable(output_bias_dist, name="Output_Bias")

            #Add the output layer
            layers.append((output_weight_var,output_bias_var))


        print("Built weights and biases")


        #Define the current Q values
        current_states = tf.placeholder(tf.float32, [None, input_width], name="Current_States")
        next_states = tf.placeholder(tf.float32, [None, input_width], name="Next_States")


        #Build the network from the layers list skip the first 
        #last since they are done seperately. 
        print("Building Network 1")
        with tf.name_scope('Relu_Layers_Network_1') as scope:
            network = tf.nn.relu(tf.matmul(current_states, layers[0][0]) + layers[0][1], name="Input_RELU")
            print("   Built Input Layer")

            layer_num = 0
            for weight, bias in layers[1:-1]:
                layer_name = "Hidden_Layer_{0}_RELU".format(layer_num)
                network = tf.nn.relu(tf.matmul(network, weight) + bias, name=layer_name)
                print("   Built Hidden Layer")
                layer_num += 1

            network = tf.squeeze(tf.matmul(network,layers[-1][0]) + layers[-1][1])
            print("   Built Output Layer")


        #Build a seccond network to run double Q
        print("Building Network 2")
        with tf.name_scope('Relu_Layers_Network_2') as scope:
            network2 = tf.nn.relu(tf.matmul(next_states, layers[0][0]) + layers[0][1], name="Input_RELU")
            print("   Built Input Layer")

            layer_num = 0
            for weight, bias in layers[1:-1]:
                layer_name = "Hidden_Layer_{0}_RELU".format(layer_num)
                network2 = tf.nn.relu(tf.matmul(network2, weight) + bias, name=layer_name)
                print("   Built Hidden Layer")
                layer_num += 1

            network2 = tf.squeeze(tf.matmul(network2,layers[-1][0]) + layers[-1][1])
            print("   Built Output Layer")

        #Get the output Q
        with tf.name_scope('Loss_Function') as scope:

            with tf.name_scope('Current_State_Outputs') as scope:
                current_Q_out_mask = tf.placeholder(tf.float32, [None, output_width])
                current_Q_out = tf.reduce_sum(tf.mul(network, current_Q_out_mask), reduction_indices=1)

            #Get the output of Q next
            with tf.name_scope('Next_State_Outputs') as scope:
                prev_rewards = tf.placeholder(tf.float32, [None, ])
                next_Q_out = prev_rewards + gamma * tf.reduce_max(network2, reduction_indices=1)

            #Create the loss function
            with tf.name_scope('Loss_Function') as scope:
                self.loss_function = tf.reduce_mean(tf.square(current_Q_out - next_Q_out))


        #Create a training function
        with tf.name_scope('Training_Function') as scope:
            self.training_function = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_function)

        #Save values for later
        self.current_states = current_states
        self.current_Q_out_mask = current_Q_out_mask
        self.prev_rewards = prev_rewards
        self.network = network



		

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    learner = deep_learner(env.observation_space, env.action_space)