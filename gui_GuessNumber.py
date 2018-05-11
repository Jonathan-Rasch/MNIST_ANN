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
        root.title("NN number predict.")
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
        #graph
        self.graph = tk.Canvas(root,width=28*15,height=14*15)
        self.graph.grid(row=2,column=0,columnspan=2)
        self.figureHandle = None # needed to keep reference to photo, will disappear otherwise
        self.draw_figure()
        #tensorflow model
        self.X = tf.placeholder(tf.float32, shape=[None, 784], name="inputs")  # inputs
        self.W = tf.Variable(initial_value=tf.random_normal(shape=[784, 10]), name="weights")  # weights
        self.b = tf.Variable(initial_value=tf.random_normal(shape=[10]), name="bias")
        self.y = tf.nn.softmax(tf.matmul(self.X, self.W) + self.b)
        saver = tf.train.Saver()
        self.session = tf.Session()
        saver.restore(self.session,"MODEL/model.ckpt")

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
            self.map[list_index] += 0.5
            if (self.map[list_index]) >= 1: self.map[cell] = 1

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
            self.draw_figure(self.normalisePred(self.session.run(fetches=self.y, feed_dict={self.X: np.array(self.map).reshape([-1,784])})))
            self.drawCanvas()
        self.drawingMode = not self.drawingMode

    def clear(self):
        # drawing map
        self.map = []
        for index in range(0, 784):
            self.map.append(0)
        self.drawingMode = False
        self.drawCanvas()
        self.draw_figure()

    def draw_figure(self,data=[1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10], loc=(0, 0)):
        # determining pred
        max = 0
        pred_num = 0
        for index,val in enumerate(data):
            if(val>max):
                max = val
                pred_num = index
        self.predLabel.configure(text="PREDICTION: {} ".format(pred_num))
        # plotting
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
