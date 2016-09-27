import gym
import tensorflow as tf
 
sess = tf.InteractiveSession()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


#Input
x = tf.placeholder(tf.float32, shape=[1, 4], name="in")
y_ = tf.placeholder(tf.float32, shape=[1, 3])


#Densely Connected Layer
W_fc1 = weight_variable([4,40])
b_fc1 = bias_variable([40])
h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

#Densely Connected Layer
W_fc2 = weight_variable([40,40])
b_fc2 = bias_variable([40])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1,W_fc2) + b_fc1)


#Readout Layer
W_fc3 = weight_variable([40, 3])
b_fc3 = bias_variable([3])
y_conv=tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)


#Train and test
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess.run(tf.initialize_all_variables())

merged_summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('logs/',sess.graph)



#for i in range(1000):
#  batch = mnist.train.next_batch(50)
#  if i%10 == 0:
#    train_accuracy = accuracy.eval(feed_dict={
#        x:batch[0], y_: batch[1], keep_prob: 1.0})
#    print("step %d, training accuracy %g"%(i, train_accuracy))
#  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#
#print("test accuracy %g"%accuracy.eval(feed_dict={
#    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

