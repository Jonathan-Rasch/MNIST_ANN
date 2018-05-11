import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
########################################################################################################################
# loading data set
########################################################################################################################
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
#plt.imshow(mnist.train.images[1].reshape(28,28),cmap="gist_gray")

########################################################################################################################
# create placeholders and vairables
########################################################################################################################
X = tf.placeholder(tf.float32,shape=[None,784],name="inputs") # inputs
    # None: first dimension is none, because this will be set later (it is the batch size)
    # 784: img has size 28*28, hence input tensor has that dimension
W = tf.Variable(initial_value=tf.zeros(shape=[784,10]),name="weights") # weights
    # weight vector, 784 inputs, 10 neurons
b = tf.Variable(initial_value=tf.zeros(shape=[10]),name="bias")
    # bias term for 10 neurons
########################################################################################################################
#create graph operations
########################################################################################################################
y = tf.matmul(X,W) #+ b
########################################################################################################################
# Loss function
########################################################################################################################
y_true = tf.nn.tanh(tf.placeholder(tf.float32,[None,10]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y))
########################################################################################################################
# Optimiser
########################################################################################################################
optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimiser.minimize(cross_entropy)
########################################################################################################################
# evaluate
########################################################################################################################
correct_pred = tf.equal(tf.argmax(y,axis=1), tf.argmax(y_true,axis=1))
    # evaluation node, returns matrix of bool where label and prediction match
evaluate = tf.reduce_mean(tf.cast(correct_pred, tf.float32))# this is another seperate graph !!
########################################################################################################################
# graphs
########################################################################################################################
# input data graph
columns = 4
rows = 4
fig_inputs=plt.figure(figsize=(8, 8))
# accuracy
fig_accuracy = plt.figure(figsize=(8,8))
fig_accuracy_x_axis = []
fig_accuracy_y_axis = []
########################################################################################################################
# Saver for saving model
########################################################################################################################
saver = tf.train.Saver()
########################################################################################################################
# session
########################################################################################################################
init = tf.global_variables_initializer()
step_before_plot = 1
step_counter = 0
prev_weight_vec = [None,None,None,None,None,None,None,None,None,None]
batch_size = 2
with tf.Session() as sess:
    sess.run(init) # initialising global variable
    for step in range(1000000):
        step_counter += 1
        batch_x,batch_y = mnist.train.next_batch(batch_size=batch_size) # this is usually hard
        sess.run(fetches=train, feed_dict={X: batch_x, y_true: batch_y})#training
        ########################################################################################################################
        # eval and Plotting
        ########################################################################################################################
        if (step_counter < step_before_plot):
            continue
        else:
            step_counter = 0
            step_before_plot += 1

        accScore = sess.run(fetches=evaluate, feed_dict={X: mnist.test.images, y_true: mnist.test.labels})
        print("AVG ACCURACY: {} %".format(round(accScore * 100, 2)))
        #input data
        plt.figure(1)
        fig_inputs.clear()
        fig_inputs.suptitle("STEP:{}, Batch_size:{}, input_images_per_plot_update:{}".format(step,batch_size,batch_size*step_before_plot))
        fig_inputs.set_facecolor('gray')
        #print(sess.run(fetches=y,feed_dict={X: batch_x[0]})) # printing probabilities
        for i in range(0, 10):
            axes = fig_inputs.add_subplot(4, 5, i+1)
            axes.set_title(str(i))
            # extracting the column for neuron i from the weight Tensor
            weights_for_neuron_i = W[0:784,i:i+1]
            weight_img = tf.reshape(tensor=weights_for_neuron_i,shape=[28,28])
            curr_weight_vec = sess.run(fetches=weight_img)
            axes.set_yticks([])
            axes.set_xticks([])
            plt.imshow(curr_weight_vec, cmap='inferno')
            if(prev_weight_vec[i] is None):
                prev_weight_vec[i] = curr_weight_vec
            delta_vec = curr_weight_vec - prev_weight_vec[i]
            prev_weight_vec[i] = curr_weight_vec
            # plotting change in weight vector
            axes = fig_inputs.add_subplot(4, 5, i + 11)
            axes.set_title("delta_"+str(i))
            axes.set_yticks([])
            axes.set_xticks([])
            plt.imshow(delta_vec, cmap='viridis')
        plt.pause(0.05)
        # accuracy graph
        plt.figure(2)
        fig_accuracy_y_axis.append(accScore)
        fig_accuracy_x_axis.append(step)
        fig_accuracy.clear()
        acc_axes = fig_accuracy.add_subplot(111)
        acc_axes.set_title("Average accuracy (in test set) vs training steps")
        acc_axes.set_ylim(bottom=0, top=1,auto=True)
        acc_axes.set_xlabel("training step ({})".format(step))
        acc_axes.set_ylabel("Average accuracy ({}%)".format(round(accScore*100,2)))
        acc_axes.set_xlim(left=0,auto=True)
        acc_line, = acc_axes.plot(fig_accuracy_x_axis, fig_accuracy_y_axis)
        plt.pause(0.05)
        if(step == 0):
            plt.pause(10)
            input("press any key to continue")

    ########################################################################################################################
    # saving
    ########################################################################################################################
    #save_path = saver.save(sess, "MODEL/model.ckpt")
    #print("Model saved in path: %s" % save_path)


