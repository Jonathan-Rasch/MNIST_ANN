import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
import skimage.measure
import skimage.filters

########################################################################################################################
# loading the dataset
########################################################################################################################
mnist_data = input_data.read_data_sets('MNIST_data/',one_hot=True)

########################################################################################################################
# utility functions
########################################################################################################################

def initialiseWeights(shape,name):
    init_random_dist = tf.random_normal(shape,mean=0,stddev=0.01)
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
# graphs
########################################################################################################################
# input data graph
columns = 4
rows = 4
fig_inputs=plt.figure(figsize=(5, 8))
# accuracy
fig_accuracy = plt.figure(figsize=(8,8))
fig_accuracy_x_axis = []
fig_accuracy_y_axis = []
fig_kernels_conv1 = plt.figure(figsize=(4,8))
gs = gridspec.GridSpec(8,4)
fig_kernels_conv2 = plt.figure(figsize=(6,8))
gs2 = gridspec.GridSpec(8,6)
correct_pred = tf.equal(tf.argmax(output_layer,axis=1), tf.argmax(y_true,axis=1))
########################################################################################################################
# TRAINING THE MODEL
########################################################################################################################
saver = tf.train.Saver()
prev_accuracy = 0
accuracy = 0
with tf.Session() as sess:
    try:
        saver.restore(sess, "MODEL_6/model.ckpt")
        sess.run(fetches=tf.local_variables_initializer())
    except:
        sess.run(fetches=tf.global_variables_initializer())
    MAX_STEP = 1000000
    for step in range(0,MAX_STEP):
        x_train,y_train = mnist_data.train.next_batch(batch_size=16) # friends dont let friends use large mini batches (we want random gradient to escape local minima !)
        sess.run(fetches=train, feed_dict={x: x_train, y_true: y_train, hold_probability: 0.65})
        if (step%5 == 0):
            print("---------------------------------------------------------------------")
            correct_on_training_set = sess.run(fetches=correct_pred, feed_dict={x: x_train, y_true: y_train, hold_probability: 0.65})
            print("STEP {}".format(step))
            matches = tf.equal(tf.argmax(output_layer,axis=1),tf.argmax(y_true,axis=1))
            accuracy = tf.reduce_mean(tf.cast(matches,tf.float32))
            batch_x, batch_y = mnist_data.train.next_batch(batch_size=1000)
            acc_val = sess.run(accuracy,feed_dict={x:batch_x,y_true:batch_y,hold_probability:1})
            print("ACCURACY: {}".format(acc_val))
            ########################################################################################################################
            # plotting the features
            ########################################################################################################################
            # input data
            if True:
                plt.figure(1)
                fig_inputs.clear()
                fig_inputs.suptitle("Input image batch")
                fig_inputs.set_facecolor('gray')
                columns = 4
                rows = 4
                for i in range(1, 1 + columns * rows):
                    axes = fig_inputs.add_subplot(rows, columns, i)
                    if (correct_on_training_set[i - 1]):
                        plt.imshow(batch_x[i - 1].reshape(28, 28), cmap="Greens")
                    else:
                        plt.imshow(batch_x[i - 1].reshape(28, 28), cmap="Reds")
                    axes.set_yticks([])
                    axes.set_xticks([])

                plt.pause(0.05)
                # accuracy graph
            if True:
                plt.figure(2)
                fig_accuracy_y_axis.append(acc_val)
                fig_accuracy_x_axis.append(step)
                fig_accuracy.clear()
                acc_axes = fig_accuracy.add_subplot(111)
                acc_axes.set_title("Average accuracy (in test set) vs training steps")
                acc_axes.set_ylim(bottom=0, top=1, auto=True)
                acc_axes.set_xlabel("training step ({})".format(step))
                acc_axes.set_ylabel("Average accuracy ({}%)".format(round(acc_val * 100, 2)))
                acc_axes.set_xlim(left=0, auto=True)
                acc_line, = acc_axes.plot(fig_accuracy_x_axis, fig_accuracy_y_axis)
                if (step == 0):
                    pass
                    plt.pause(10)
                    input("press any key to continue")
                plt.pause(0.05)

            ########################################################################################################################
            #for later TODO
            ########################################################################################################################
            #gr = tf.get_default_graph()
            #conv1_kernel = gr.get_tensor_by_name('Variable_2/read').eval()
            feature_weights = np.array(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'conv1:0')))
            # extracting kernels.
            conv1_kernels = []
            for i in range(0,32):
                conv1_kernels.append([])
            # for row_index in range(0,5):
            #     for column_index in range(0,5):
            for filter_index in range(0,32):
                weight_vec = feature_weights[0, 0:10, 0:10, 0, filter_index]
                conv1_kernels[filter_index].append(weight_vec)
            plt.figure(3)
            fig_kernels_conv1.clear()
            fig_kernels_conv1.suptitle("Convolution 1 kernels")
            fig_kernels_conv1.set_facecolor('gray')
            columns = 4
            rows = 8
            gs.update(wspace=0.025, hspace=0.025)
            for i in range(0, columns * rows):
                axes = plt.subplot(gs[i])
                plt.axis('on')
                kernel_img = conv1_kernels[i][0]
                plt.imshow(kernel_img)
                axes.set_yticks([])
                axes.set_xticks([])
                axes.set_aspect('equal')
            plt.pause(0.05)
            ########################################################################################################################
            # convolution 2 layer
            ########################################################################################################################
            feature_weights = np.array(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'conv2:0')))
            conv2_kernels = []
            for kernel2_index in range(0,64):
                weight_vec_with_inputs = feature_weights[0,0:7,0:7,0:32,kernel2_index].reshape(1,7,7,32)
                conv1_weighted_kernel = np.zeros(shape=(10,10))
                kernel_conv2_section = np.zeros(shape=(70, 70)) # one 1x1 (10x10) section of the 7x7 (70x70) kernel
                for row_ind,column_ind,filter_ind in itertools.product(range(7),range(7),range(32)):
                    # multiplying the each kernel from each filter from conv1 with the weights of conv2 for that kernel/filter
                    weight_for_kernel = weight_vec_with_inputs[0,row_ind,column_ind,filter_ind]
                    kernel = conv1_kernels[filter_ind][0]
                    #pooled_kernel = skimage.measure.block_reduce(kernel, block_size=(2, 2),func=np.max)
                    conv1_weighted_kernel += kernel * weight_for_kernel
                    if(filter_ind == 31):
                        row_start = row_ind*10
                        row_end = (row_ind+1)*10
                        col_start = column_ind*10
                        col_end = (column_ind+1)*10
                        kernel_conv2_section[row_start:row_end, col_start:col_end] = conv1_weighted_kernel
                        conv1_weighted_kernel = np.zeros(shape=(10, 10))
                conv2_kernels.append(skimage.filters.gaussian(image=kernel_conv2_section,mode='nearest',preserve_range=True,sigma=(5,5)))
            #plotting
            plt.figure(4)
            fig_kernels_conv2.clear()
            fig_kernels_conv2.suptitle("Convolution 2 kernels")
            fig_kernels_conv2.set_facecolor('gray')
            columns = 6
            rows = 8
            gs.update(wspace=0.025, hspace=0.025)
            for i in range(0, columns * rows):
                axes = plt.subplot(gs2[i])
                plt.axis('on')
                kernel_img = conv2_kernels[i]
                plt.imshow(kernel_img)
                axes.set_yticks([])
                axes.set_xticks([])
                axes.set_aspect('equal')
            plt.pause(0.05)

        if (step%500 == 0 and step > 0):# float(acc_val) > prev_accuracy and step > 1):
            prev_accuracy = float(acc_val)
            print("SAVING MODEL...")
            save_path = saver.save(sess, "MODEL_6/model.ckpt")