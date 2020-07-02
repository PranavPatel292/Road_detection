
# Author: Evan Juras

#####################we have taken the author's code as reference code and modify it depandig on our requirement.########################

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

import tkinter as tk
from tkinter import *
from tkinter.ttk import *
import PIL
from PIL import Image, ImageTk
import googlemaps #use for the Google MAP API



import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util


pb_file_ssd = "C:/Users/Pranav Patel/Documents/DeepNeuralnet/Road_Work_Videos/interface_graph/frozen_inference_graph_ssd.pb"


label_map_ssd = "C:/Users/Pranav Patel/Documents/DeepNeuralnet/Road_Work_Videos/interface_graph/label_map_ssd.pbtxt"
# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = pb_file_rcnn



#C:/Users/Pranav Patel/Documents/DeepNeuralnet/Road_Work_Videos/interface_graph/Vaibhav/Assignemnt_3_Final_code/frozen_inference_graph.pb"

# Path to label map file
PATH_TO_LABELS = label_map_rcnn

# Path to video
PATH_TO_VIDEO = "C:/Users/Pranav Patel/Documents/DeepNeuralnet/Road_Work_Videos/short_video_3.mp4"

# Number of classes the object detector can identify
NUM_CLASSES = 7 #7 for rncc


maxWidth = 720
maxHeight = 1280

mainWindow = tk.Tk()
mainWindow.geometry('1920x1080')
mainWindow.resizable(0,0)
# video frame
mainFrame = Frame(mainWindow)
mainFrame.place(x=20, y=20)
subFrame = Frame(mainWindow)
subFrame.place(x = 860, y = 50)

lblmain = tk.Label(mainFrame)
lblmain.grid(row=0, column=0)


video = cv2.VideoCapture(PATH_TO_VIDEO)

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)

counter_road_work = 0


#method for making the Video apper ont custom made GUI version
def show_frame():
	global counter_road_work
	ret, frame = video.read()
	cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame_expanded = np.expand_dims(cv2image, axis=0)


	(boxes, scores, classes, num) = sess.run(
		[detection_boxes, detection_scores, detection_classes, num_detections],
		feed_dict={image_tensor: frame_expanded})

	vis_util.visualize_boxes_and_labels_on_image_array(
		frame,
		np.squeeze(boxes),
		np.squeeze(classes).astype(np.int32),
		np.squeeze(scores),
		category_index,
		use_normalized_coordinates=True,
		line_thickness=8,
		min_score_thresh=0.95)

	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	for i,b in enumerate(boxes[0]):
		 if classes[0][i] == 5 and scores[0][i] >= 0.95: #5 for RNCC #this if cluse is for detecting the most precise detection and then to call the Google Map API (which is not implement in this version)
		 	counter_road_work += 1
		 	change_text("Road Work detected\n")
		 	##print("Road work detected")
		 if counter_road_work > 15:
		 	counter_road_work = 0
		 	latitude = 33.883167
		 	longitute = 151.200344
		 	string_name = "Making Marker to Google Map with Latitude: ", latitude, "and Longitute: ", longitute, "\n"
		 	change_text(string_name)
		 	##print("Making Marker to Google Map with Latitude: ", latitude, "and Longitute: ", longitute)

	img = Image.fromarray(frame).resize((720,1280))

	imgtk = ImageTk.PhotoImage(image = img)
	lblmain.imgtk = imgtk
	lblmain.configure(image = imgtk)
	lblmain.after(10, show_frame) #every 10 milisecond the new frame is taken


#to exit the loop or GUI
def exitLoop():
	exit()

S = tk.Scrollbar(subFrame)
T = tk.Text(subFrame, height=50, width=50)

#to append the text in the scroll bar.!
def change_text(quote):
	T.insert(tk.END, quote)

#for adding the scroll bar to see what model has predicated.!

def add_scroll():
	S.pack(side=tk.RIGHT, fill=tk.Y)
	T.pack(side=tk.LEFT, fill=tk.Y)
	S.config(command=T.yview)
	T.config(yscrollcommand=S.set)

#for adding the button on the GUI
def add_button():
	btn2 = tk.Button(subFrame, text="Exit", command=exitLoop).pack(side=tk.BOTTOM)


#calling the function, just like a main function in C language;
show_frame()
add_button()
add_scroll()
mainWindow.mainloop()

