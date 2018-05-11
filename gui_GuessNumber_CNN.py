import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib as mpl
import numpy as np
import tensorflow as tf
import math

SIZE_UNIT = 15

class GUI:
    def __init__(self,root):
        self.root = root
        # setting title and layout
        root.title("CNN number predict.")
        root.geometry("420x{}".format(14*15+28*15))
        root.resizable(False, False)
        # drawing map
        self.lastPosition = -1
        self.map = []
        for index in range(0, 784):
            self.map.append(0)
        #creating buttons
        self.clearButton = tk.Button(root,text="Clear Canvas",command=self.clear,width=25)
        self.clearButton.grid(row=0,column=0,sticky=tk.W)
        # prediction label
        self.predLabel = tk.Label(root,text= "PREDICTION: -- ")
        self.predLabel.grid(row=0,column=1,sticky=tk.W)
        #Canvas
        self.canvas = tk.Canvas(root, width =28 * SIZE_UNIT, height=28 * SIZE_UNIT)
        self.canvas.grid(row=1, column=0,columnspan=2)
        self.drawCanvas()
        self.drawingMode = False
        self.canvas.bind('<Motion>',self.mmove)
        self.canvas.bind("<Button-1>", self.mclick)
        self.calls_to_draw = 0
        #graph
        self.graph = tk.Canvas(root,width=28*15,height=14*15)
        self.graph.grid(row=2,column=0,columnspan=2)
        self.figureHandle = None # needed to keep reference to photo, will disappear otherwise
        self.draw_figure()
        #tensorflow model
        self.saver = tf.train.Saver
        ########################################################################################################################
        # utility functions
        ########################################################################################################################

        def initialiseWeights(shape, name):
            init_random_dist = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial_value=init_random_dist, name=name)

        def initialiseBias(shape):
            init_bias_values = tf.constant(0.1, shape=shape)
            return tf.Variable(initial_value=init_bias_values)

        def convo2d(inputTensor, convolutionKernel):
            '''
            :param inputTensor: of shape [batch,height,width,channel]
            :param convolutionKernel: [height,width,channel in, channels out]
            :return:
            '''
            return tf.nn.conv2d(input=inputTensor, filter=convolutionKernel, padding="SAME", strides=[1, 1, 1, 1])

        def max_pool_2x2(inputTensor):
            # inputTensor --> [Batch,Height,Width,Channel]
            return tf.nn.max_pool(value=inputTensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        def convolutionalLayer(inputTensor, shape, name):
            # inputTensor --> [batch,height,width,channel]
            weights = initialiseWeights(shape, name=name)
            bias = initialiseBias([shape[3]])
            return tf.nn.relu(convo2d(inputTensor, weights) + bias)

        def denseLayer(inputTensor, size, name):
            in_size = int(inputTensor.get_shape()[1])  # 0th element is batch
            weights = initialiseWeights([in_size, size], name=name)
            bias = initialiseBias([size])
            return tf.matmul(inputTensor, weights) + bias

        ########################################################################################################################
        # create placeholders
        ########################################################################################################################
        self.x = tf.placeholder(tf.float32, shape=[None, 784], name="inputs")
        ########################################################################################################################
        # Layers
        ########################################################################################################################
        self.image_input = tf.reshape(self.x, shape=[-1, 28, 28, 1])
        # convolution and pooling 1
        self.convo1_layer = convolutionalLayer(self.image_input, shape=[10, 10, 1, 32],name="conv1")
        self.pooling1_layer = max_pool_2x2(self.convo1_layer)  # size is now changed to [-1,14,14,32]
        # convolution and pooling 2
        self.convo2_layer = convolutionalLayer(self.pooling1_layer, shape=[7, 7, 32, 64],name="conv2")
        self.pooling2_layer = max_pool_2x2(self.convo2_layer)  # size is now changed to [-1,7,7,64]
        # flattening
        self.flattened_layer = tf.reshape(self.pooling2_layer, shape=[-1, 7 * 7 * 64])
        # dense layer and dropout 1
        self.dense1_layer = tf.nn.relu(denseLayer(self.flattened_layer, size=1024,name="dense1"))
        self.hold_probability = tf.placeholder(tf.float32)
        self.dense1_dropout = tf.nn.dropout(self.dense1_layer, keep_prob=self.hold_probability)
        # # dense layer and dropout 2
        # self.dense2_layer = tf.nn.relu(denseLayer(self.dense1_dropout, size=256))
        # self.dense2_dropout = tf.nn.dropout(self.dense2_layer, keep_prob=self.hold_probability)
        # output layer
        self.output_layer = denseLayer(self.dense1_dropout, size=10,name="dense2")
        self.predict = tf.nn.softmax(self.output_layer)
        ########################################################################################################################
        # loss function and optimiser
        ########################################################################################################################
        saver = tf.train.Saver()
        self.session = tf.Session()
        saver.restore(self.session, "MODEL_3/model.ckpt")
        self.session.run(fetches=tf.local_variables_initializer())

    def drawCanvas(self):
        self.canvas.create_rectangle(0, 0, 28 * SIZE_UNIT, 28 * SIZE_UNIT, fill="gray")
        for index,value in enumerate(self.map):
            y_coord = math.floor(index/28)*SIZE_UNIT
            x_coord = (index%28)*SIZE_UNIT
            color = '#%02x%02x%02x' % (int(255*value), int(255*value), int(255*value))
            self.canvas.create_rectangle(x_coord, y_coord, x_coord + SIZE_UNIT, y_coord + SIZE_UNIT, fill=color)

    def mmove(self,event):
        if(self.drawingMode):
            x_index = math.floor(event.x/SIZE_UNIT)
            y_index = math.floor(event.y/SIZE_UNIT)
            list_index = y_index*28 + x_index
            if(self.lastPosition == list_index):
                return
            self.lastPosition = list_index
            top_left = (y_index-1)*28 + (x_index-1)
            left = (y_index ) * 28 + (x_index - 1)
            bot_left = (y_index + 1) * 28 + (x_index - 1)
            top = (y_index - 1) * 28 + (x_index )
            bot = (y_index + 1) * 28 + (x_index)
            top_right = (y_index - 1) * 28 + (x_index + 1)
            right = (y_index) * 28 + (x_index + 1)
            bot_right = (y_index + 1) * 28 + (x_index + 1)
            othercells = [top_left,left,bot_left,top,bot,top_right,right,bot_right]
            for cell in othercells:
                if(cell >= 0 and cell < 784):
                    self.map[cell] += 0.15
                    if (self.map[cell]) >= 1: self.map[cell] = 1
            self.map[list_index] += 0.65
            if (self.map[list_index]) >= 1: self.map[cell] = 1
            self.calls_to_draw += 1

    def normalisePred(self,predData):
        predData = predData[0]
        max = None
        min = None
        for num in predData:
            if(max == None or num > max):
                max = num
            if(min == None or num < min):
                min = num
        max -= min
        for index,num in enumerate(predData):
            num -= min
            num /= max
            predData[index] = num
        return predData

    def mclick(self,event):
        if(self.drawingMode):
            self.draw_figure(self.session.run(fetches=self.predict, feed_dict={self.x: np.array(self.map).reshape([-1,784]),self.hold_probability:1.0}).tolist()[0])
            self.drawCanvas()
        self.drawingMode = not self.drawingMode

    def clear(self):
        # drawing map
        self.map = []
        for index in range(0, 784):
            self.map.append(0)
        self.drawingMode = False
        self.drawCanvas()

    def draw_figure(self,data=[1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10], loc=(0, 0)):
        # determining pred
        max = 0
        pred_num = 0
        for index, val in enumerate(data):
            if (val > max):
                max = val
                pred_num = index
        self.predLabel.configure(text="PREDICTION: {} ".format(pred_num))
        # plot
        canvas = self.graph
        figure = mpl.figure.Figure(figsize=(4.2, 1.8))
        figure_canvas_agg = FigureCanvasAgg(figure)
        ax = figure.add_subplot(111)
        figure.subplots_adjust(left=0.1, bottom=0.3, right=None, top=None, wspace=None, hspace=None)
        ax.set_ylim(bottom=0,top=1)
        figure.suptitle('prediction category')
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([0,1,2,3,4,5,6,7,8,9])
        index = np.arange(10)
        rects1 = ax.bar(index,data,color='red',label='certainty')
        ax.set_xlabel('Category')
        ax.set_ylabel('Certainty')
        figure_canvas_agg.draw()
        figure_x, figure_y, figure_w, figure_h = figure.bbox.bounds
        figure_w, figure_h = int(figure_w), int(figure_h)
        photo = tk.PhotoImage(master=canvas, width=figure_w-10, height=figure_h-10)
        # Position: convert from top-left anchor to center anchor
        canvas.create_image(loc[0] + (figure_w / 2), loc[1] + (figure_h / 2), image=photo)
        tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)
        # Return a handle which contains a reference to the photo object
        # which must be kept live or else the picture disappears
        self.figureHandle = photo
        return photo


if __name__ == "__main__":
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()
