import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

########################################################################################################################
# loading the dataset
########################################################################################################################
mnist_data = input_data.read_data_sets('MNIST_data/',one_hot=True)

########################################################################################################################
# utility functions
########################################################################################################################

def initialiseWeights(shape,name):
    init_random_dist = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial_value=init_random_dist,name=name)

def initialiseBias(shape):
    init_bias_values = tf.constant(0.1,shape=shape)
    return tf.Variable(initial_value=init_bias_values)

def convo2d(inputTensor,convolutionKernel):
    '''
    :param inputTensor: of shape [batch,height,width,channel]
    :param convolutionKernel: [height,width,channel in, channels out]
    :return:
    '''
    return tf.nn.conv2d(input=inputTensor,filter=convolutionKernel,padding="SAME",strides=[1,1,1,1])

def max_pool_2x2(inputTensor):
    # inputTensor --> [Batch,Height,Width,Channel]
    return tf.nn.max_pool(value=inputTensor,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

def convolutionalLayer(inputTensor,shape,name):
    # inputTensor --> [batch,height,width,channel]
    weights = initialiseWeights(shape,name=name)
    bias = initialiseBias([shape[3]])
    return tf.nn.relu(convo2d(inputTensor,weights)+bias)

def denseLayer(inputTensor,size,name):
    in_size = int(inputTensor.get_shape()[1]) # 0th element is batch
    weights = initialiseWeights([in_size,size],name=name)
    bias = initialiseBias([size])
    return tf.matmul(inputTensor,weights)+bias

########################################################################################################################
# create placeholders
########################################################################################################################
x = tf.placeholder(tf.float32,shape=[None,784],name="inputs")
y_true = tf.placeholder(tf.float32,shape=[None,10],name="labels")
########################################################################################################################
# Layers
########################################################################################################################
image_input = tf.reshape(x,shape=[-1,28,28,1])
# convolution and pooling 1
convo1_layer = convolutionalLayer(image_input,shape=[10,10,1,32],name="conv1")
pooling1_layer = max_pool_2x2(convo1_layer) # size is now changed to [-1,14,14,32]
# convolution and pooling 2
convo2_layer = convolutionalLayer(pooling1_layer,shape=[7,7,32,64],name="conv2")
pooling2_layer = max_pool_2x2(convo2_layer) # size is now changed to [-1,7,7,64]
# flattening
flattened_layer = tf.reshape(pooling2_layer,shape=[-1,7*7*64])
# dense layer and dropout 1
dense1_layer = tf.nn.relu(denseLayer(flattened_layer,size=1024,name="dense1"))
hold_probability = tf.placeholder(tf.float32)
dense1_dropout = tf.nn.dropout(dense1_layer,keep_prob=hold_probability)
# # dense layer and dropout 2
# dense2_layer = tf.nn.relu(denseLayer(dense1_dropout,size=512))
# dense2_dropout = tf.nn.dropout(dense2_layer,keep_prob=hold_probability)
# output layer
output_layer = denseLayer(dense1_dropout,size=10,name="dense2")
########################################################################################################################
# loss function and optimiser
########################################################################################################################
loss_fnc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=output_layer))
optimiser = tf.train.AdagradOptimizer(learning_rate=0.01) # create adam optimiser, (per parameter learning rate + another benefit TODO research)
train = optimiser.minimize(loss_fnc) # tel optimiser to minimize the cost function
########################################################################################################################
# TRAINING THE MODEL
########################################################################################################################
saver = tf.train.Saver()
prev_accuracy = 0
accuracy = 0
with tf.Session() as sess:
    try:
        saver.restore(sess, "MODEL_4/model.ckpt")
        sess.run(fetches=tf.local_variables_initializer())
    except:
        sess.run(fetches=tf.global_variables_initializer())
    MAX_STEP = 1000000
    for step in range(0,MAX_STEP):
        x_train,y_train = mnist_data.train.next_batch(batch_size=16) # friends dont let friends use large mini batches (we want random gradient to escape local minima !)
        sess.run(fetches=train, feed_dict={x: x_train, y_true: y_train, hold_probability: 0.65})
        if (step%1 == 0):
            print("---------------------------------------------------------------------")
            print("STEP {}".format(step))
            matches = tf.equal(tf.argmax(output_layer,axis=1),tf.argmax(y_true,axis=1))
            accuracy = tf.reduce_mean(tf.cast(matches,tf.float32))
            batch_x, batch_y = mnist_data.train.next_batch(batch_size=1000)
            acc_val = sess.run(accuracy,feed_dict={x:batch_x,y_true:batch_y,hold_probability:1})
            print("ACCURACY: {}".format(acc_val))
            ########################################################################################################################
            # plotting the features
            ########################################################################################################################
            #gr = tf.get_default_graph()
            #conv1_kernel = gr.get_tensor_by_name('Variable_2/read').eval()
            feature_weights = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'conv1:0'))
            #print(feature_weights)
        if (step%1000 == 0 and step > 0):# float(acc_val) > prev_accuracy and step > 1):
            prev_accuracy = float(acc_val)
            print("SAVING MODEL...")
            save_path = saver.save(sess, "MODEL_4/model.ckpt")