import gym
import tensorflow as tf
 
class deep_Learner():
	self.loss_function
	self.training_function
	self.current_Q

	def __init__(self,observation_space, action_space):
		self.observation_space = observation_space
		self.action_space = action_space


		#Build the network
		self.build_net(self, 
						observation_space.shape[0], 	#Input Width
						action_space.n, 				#Output Width
						2,								#Hidden Nodes
						.0001, 							#Learning Rate
						.99)  							#Gamma


		#Create and ititilize session
		self.session = tf.InteractiveSession()
		self.session.run(tf.initialize_all_variables())


	def build_net(self,input_width,output_width, num_of_hidden_layers, learning_rate, gamma):
		layers = []

		#Add the input layer
		layers.append(tf.Variable(tf.random_uniform([input_width, 128])),
						tf.Variable(tf.random_uniform([128], -1.0, 1.0)))

		#Add <num_of_hidden_layers> extra layers
		for layer_number in num_of_hidden_layers:
			layer.append((
				tf.Variable(tf.random_uniform([input_width, 128])),
				tf.Variable(tf.random_uniform([128], -1.0, 1.0))))


		#Add the input layer
		layers.append(tf.Variable(tf.random_uniform([128, output_width])),
						tf.Variable(tf.random_uniform([output_width], -1.0, 1.0)))


		#Define the current Q values
		current_Q = tf.placeholder(tf.float32, [None, input_width])
		next_Q = tf.placeholder(tf.float32, [None, output_width])


		#Build the network from the layers list
		network = tf.nn.relu(tf.matmul(current_Q, layers[0][0]) + layers[0][1])
		for weight, bias in layers[1:]
			network = tf.nn.relu(tf.matmul(network, weight) + bias)
		network = tf.squeeze(tf.matmul(network,layers[-1][0],layers[-1][1]))

		#Build a seccond network to run double Q
		network2 = tf.nn.relu(tf.matmul(current_Q, layers[0][0]) + layers[0][1])
		for weight, bias in layers[1:-1]
			network2 = tf.nn.relu(tf.matmul(network, weight) + bias)
		network2 = tf.squeeze(tf.matmul(network2,layers[-1][0],layers[-1][1]))
		
		#Get the output Q
		current_Q_out_mask = tf.placeholder(tf.float32, [None, output_width])
		current_Q_out = tf.reduce_sum(tf.mul(network, current_Q_out_mask), reduction_indices=1)
		
		#Get the output of Q next
		prev_rewards = tf.placeholder(tf.float32, [None, ])
		next_Q_out = prev_rewards + gamma * tf.reduce_max(network2, reduction_indices=1)

		#Create ans save the loss and training functions
        self.loss_function = tf.reduce_mean(tf.square(current_Q_out - next_Q_out))
		self.training_function = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

		#Save values for later
		self.current_Q = current_Q
		self.current_Q_out_mask = current_Q_out_mask
		self.prev_rewards = prev_rewards
		self.next_Q = next_Q
		self.network = network



		


