# The trained model for mask detection deployed locally using the machine's webcam
# ================================================================================
# This script is for a model stored in an HDF5 format

### Import needed modules

# Basic packages needed for data analysis, visualization and manipulation
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import cv2

# Tensorflow packages for constructing CNN models
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization, Input, GlobalAveragePooling2D
from tensorflow.keras import Sequential, regularizers, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam 
from math import ceil

### Instantiate the model

# Load model from HDF5 file
model = load_model("CNN_model.h5")

### Prepare the webcam stream

# The labels and colors of the bounding box for the presence of masks
# Dict where keys are the prediction results
outcomes = {0: 'Without Mask', 1: 'Mask'} # Labels
box_dict = {0: (0,0,255), 1: (0,255,0)} # Colors in a BGR format

# The compression factor for the frames read from the webcam
rect_size = 4

### Start the webcam stream 

cap = cv2.VideoCapture(0) # Start stream
haarcascade = cv2.CascadeClassifier("<Path to the Haar's Cascade Classifier XML file>")
# `haarcascade` is a model that comes with OpenCV (cv2) for detecting faces
# The file path for loading the CascadeClassifier is typically -> 
# os.path.dirname(cv2.__file__) + '/data/haarcascade_frontalface_default.xml'

### Main loop that processes the frames and is ended only by pressing key `q`

while True: # Perpetual loop until key `q` is pressed
    # Captures frame
    (rval, im) = cap.read() # Reads a frame as `im` from the video stream 
    im = cv2.flip(im, 1, 1) # Flips frame 
    
    # Gets regions of interest that contain faces
    rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size)) 
    # Resizing by the scale of `rect_size`
    faces = haarcascade.detectMultiScale(rerect_size)
    # Gets coordinates of regions of interest using `haarcascade`

    ### Iterate through every set of coordinates for each region of interest in the current frame

    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f] # Resizing back to original size
        
        # Extracting region of interest from current frame using coordinates
        face_img = im[y:y+h, x:x+w]
        face_img_color = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB) 
        # Flip color channels as model is trained on RGB channels but stream is in BGR format
        rerect_sized = cv2.resize(face_img_color, (224, 224)) # Resize to model input size
        normalized = rerect_sized / 255.0 # Rescale each pixel to a value between 0 and 1
        
        # Package the np.arrays properly
        reshaped = np.reshape(normalized, (1, 224, 224, 3)) 
        reshaped = np.vstack([reshaped]) 
        
        ### Predicting presence of mask
        result = model.predict(reshaped) # Mask Detection CNN Model used
        
        # Processing the predictions
        pred_number = str(round(result[0][0], 2)) # Prediction rounded to 2 d.p.
        label = (result < 0.5).astype('int32')[0][0] # The class predicted represented as 0 or 1
        
        # Setting the bounding box and other labels
        cv2.rectangle(im, (x, y), (x+w, y+h), box_dict[label], 2) # Main box
        cv2.rectangle(im, (x, y-40), (x+w, y), box_dict[label], -1) # Secondary box for label
        cv2.putText(im, (pred_number + " - " + outcomes[label]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        # Label text on the secondary box
    
    ### Display all the faces with the corresponding boxes and labels in the current frame
    cv2.imshow('LIVE', im) # `LIVE` is the name of the video stream window
    
    # Exit the while-loop if key `q` is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

### End video stream and close all windows

cap.release() # End video stream
cv2.destroyAllWindows() # Close all windows