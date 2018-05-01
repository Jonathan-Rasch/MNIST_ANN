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
W = tf.Variable(initial_value=tf.random_normal(shape=[784,10]),name="weights") # weights
    # weight vector, 784 inputs, 10 neurons
b = tf.Variable(initial_value=tf.random_normal(shape=[10]),name="bias")
    # bias term for 10 neurons
########################################################################################################################
#create graph operations
########################################################################################################################
y = tf.matmul(X,W) + b
########################################################################################################################
# Loss function
########################################################################################################################
y_true = tf.placeholder(tf.float32,[None,10])
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
with tf.Session() as sess:
    sess.run(init) # initialising global variables
    for step in range(1000):
        batch_x,batch_y = mnist.train.next_batch(batch_size=16) # this is usually hard
        correct_on_training_set = sess.run(fetches=correct_pred, feed_dict={X: batch_x, y_true: batch_y})
        sess.run(fetches=train, feed_dict={X: batch_x, y_true: batch_y})#training
        accScore = sess.run(fetches=evaluate, feed_dict={X: mnist.test.images, y_true: mnist.test.labels})
        print("AVG ACCURACY: {} %".format(round(accScore* 100,2)))

        ########################################################################################################################
        # Plotting
        ########################################################################################################################
        #input data
        plt.figure(1)
        fig_inputs.clear()
        fig_inputs.suptitle("Input image batch")
        fig_inputs.set_facecolor('gray')
        #print(sess.run(fetches=y,feed_dict={X: batch_x[0]})) # printing probabilities
        for i in range(1, 1 + columns * rows):
            axes = fig_inputs.add_subplot(rows, columns, i)
            if(correct_on_training_set[i-1]):
                plt.imshow(batch_x[i - 1].reshape(28, 28), cmap="Greens")
            else:
                plt.imshow(batch_x[i - 1].reshape(28, 28), cmap="Reds")
            axes.set_yticks([])
            axes.set_xticks([])

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
    ########################################################################################################################
    # saving
    ########################################################################################################################
    save_path = saver.save(sess, "MODEL_2/model.ckpt")
    print("Model saved in path: %s" % save_path)


