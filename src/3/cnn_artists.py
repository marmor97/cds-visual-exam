'''
Multi-class classification of impressionist painters

This project deals with a multi-class classification of impressionist painters with data from Kaggle (https://www.kaggle.com/delayedkarma/impressionist-classifier-data) and applies both a non-pretrained and pre-trained model (VGG16) to the data. 

'''

# Data tools for data wrangling and path modifications
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from contextlib import redirect_stdout

# Command line tools
import argparse

# Image transformation tools
import cv2

# Machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# Tensorflow tools
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)

from tensorflow.keras.utils import plot_model 
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# Layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout)
import tensorflow.keras.models
from tensorflow.keras.models import Model

# Optimizers
from tensorflow.keras.optimizers import SGD, Adam

# Plotting
import matplotlib.pyplot as plt

# Sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

class cnn_artists():
    '''This is a class for performing a Convolutional Neural Network classification on a Kaggle dataset with impressionist painters (data can be found here https://www.kaggle.com/delayedkarma/impressionist-classifier-data)
    '''
    def __init__(self, args):
        self.args = args
        
    def data_preparation(self, folder, save_labels = None):   
        ''' Function for loading data from train and validation folders, resizing pictures, appending these to arrays and normalizing values.
        '''
        
        print("[INFO] Preparing data...")
        

        # Empty list for labels and for images
        if save_labels == True:
            labelNames = []
        X = []
        y = []
        # Definition of label binarizer
        lb = LabelBinarizer()

        # Train data
        for subfolder in Path(folder).glob("*"): 
            artist = os.path.basename(subfolder) # Only keeping last part og path
            if save_labels == True: # If true it save one instance of each artist
                labelNames.append(artist) # Appends to list above
            # Take the current subfolder
            for pic in Path(subfolder).glob("*.jpg"): # Taking all elements in the folder
                # Read image
                pic_array = cv2.imread(str(pic)) 
                # Resize image
                compressed = cv2.resize(pic_array, 
                                       (self.args.resize, 
                                        self.args.resize), 
                                        interpolation = cv2.INTER_AREA) 
                X.append(compressed)
                y.append(artist)
        
        if save_labels == True:
            self.labelNames=labelNames
        
        # Making it arrays 
        X = np.array(X) 
        y = np.array(y)

        # Normalization
        X = X.astype("float") / 255.

        # Label binarization
        # One-hot encoding
        y = lb.fit_transform(y)

        return X, y
    
    
    def model_preparation(self, trainX = None, trainY = None, testX = None, testY = None):
        '''
        Function for setting up model architecture and model fitting. Can both be pre-trained and non-pretrained model dependent on the 'classifier' argument specified in the terminal.
        '''
        print("[INFO] Preparing and fitting model...")

        
        if self.args.classifier == 'non-pretrained':
            # Define model
            # initialise model
            self.model = Sequential()

            # define CONV => RELU layer
            self.model.add(Conv2D(32, (3, 3), # 32 = neurons, (3,3) = kernel size
                             padding="same", # Adding a padded layer of 0 
                             input_shape=(self.args.resize, self.args.resize, 3)))
            self.model.add(Activation("relu"))
            
            # Softmax classifier
            self.model.add(Flatten())
            self.model.add(Dense(10))
            self.model.add(Activation("softmax"))
            
            # Compile model
            opt = SGD(learning_rate =.01) # Learning rate 0.001 --> 0.01 are the usual values
            self.model.compile(loss="categorical_crossentropy", # Loss function also used in backpropagation networks
                          optimizer=opt, # Specifying opt
                          metrics=['accuracy'])
                        
        elif self.args.classifier == 'pretrained':
            # Load the model
            self.model = VGG16()
            # Load model without classifier layers
            self.model = VGG16(include_top=False, # load network without top layer
                          pooling='avg', # average pooling layer
                          input_shape=(self.args.resize, self.args.resize, 3)) # this is our shape which we defined with tf.keras.preprocessing.image_dataset_from_directory
            
             # Mark loaded layers as not trainable - because if they were then we retrain all at make new weights and we don't that
            for layer in self.model.layers:
                layer.trainable = False

            # Number of output classes
            num_classes = 10

            # Add new classifier layers
            flat1 = Flatten()(self.model.layers[-1].output) #the return value of Flatten() is a function apparently
            class1 = Dense(256, 
                           activation='relu')(flat1)
            
            output = Dense(num_classes, 
                           activation='softmax')(class1)

            # Define new model
            self.model = Model(inputs=self.model.inputs, # inputs 
                          outputs=output) # outputs
            
            # Compiling model
            # We compile using ADAM and categorical cross-entropy as the loss function.
            self.model.compile(optimizer=Adam(learning_rate=0.001),
                                  loss='categorical_crossentropy',
                                  metrics=['accuracy'])
        # Saving model summary in path defined from commandline (default is out)
        with open(os.path.join(self.args.outpath, f'modelsummary_{self.args.classifier}.txt'), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()
            
        # Fitting the model and saving it as self.H to plot learning curve later
        self.H = self.model.fit(trainX, trainY, 
              validation_data=(testX, testY), # The validation data, used to test at every epoch 
              batch_size=self.args.batches, # N batches
              epochs=self.args.epochs, # Epochs
              verbose=1) # Printing progress
     
                
    def model_evaluation(self, testX = None, testY = None):
        '''
        Performance evaluation function - saves a classification report containing f1 scores and accuracies for all labels. Can calculate performance on both a pre-trained and a non-pretrained model dependent on the 'classifier' argument specified in the terminal.
        '''
        print("[INFO] Evaluating model...")

        predictions = self.model.predict(testX, batch_size = self.args.batches) # Use the test set to predict the labels! With batches defined in the commandline (or default 32)

        # Comparing predictions to our test labels
        report = pd.DataFrame(classification_report(testY.argmax(axis=1), # y true - .argmax(axis=1) to retrieve actual labels
                                predictions.argmax(axis=1), # y pred .argmax(axis=1) to retrieve actual labels
                                target_names=self.labelNames, # labels defined in the first function
                                       output_dict=True)) # Dictionary to be able to transform to pd.DataFrame
            
        print(report)
        
        # Save to csv in outpath    
        report.to_csv(os.path.join(self.args.outpath, f'classification_report_{self.args.classifier}.csv'))
        print(f'[INFO] report saved in {self.args.outpath} as classification_report_{self.args.classifier}.csv')
  
    def plot_history(self): 
        '''
        Plot function to plot the learning curves of a model throughout a period of epochs. Explanation on learning curves can be found here https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/.
        '''
        print("[INFO] Saving plot of learning curves...")
        
        # Visualize performance
        plt.style.use("fivethirtyeight")
        plt.figure()
        plt.plot(np.arange(0, self.args.epochs), self.H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, self.args.epochs), self.H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, self.args.epochs), self.H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, self.args.epochs), self.H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.tight_layout()
        
 # Specifying layout
        plt.savefig(os.path.join(self.args.outpath, f'loss_accuracy_curve_{self.args.classifier}.png')) # Specifying save location and name
        plt.show()
        print(f'figure of learning curves is saved in {self.args.outpath} as loss_accuracy_curve_{self.args.classifier}.png') # Printing info on path
    
def main(): 
    ap = argparse.ArgumentParser(description="[INFO] class made to run CNN on Kaggle data set with impressionist artists") 
    
    # Classifer type
    ap.add_argument('-c',
                    '--classifier',
                    required = True, # It is required
                    help='str, type of classifier used - either non-pretrained or pretrained',
                    choices = ['non-pretrained','pretrained']) # Specifies the names that can be in the argument
    # Train folder
    ap.add_argument('-t',
                    '--train',
                    help='str, path for training data', 
                    default = os.path.join("..","..","data","3","training", "training")) 
    # Validation folder
    ap.add_argument('-val',
                    '--validation',
                    help='str, path for validation data', 
                    default = os.path.join("..","..","data","3","validation", "validation")) 
    
    ap.add_argument('-o',
                    '--outpath',
                    help='str, path for output data', 
                    default = os.path.join("..","..","out","3")) 
    
    # Image size
    ap.add_argument('-r',
                    '--resize',
                    help="int resize value of paintings",
                    type = int, # Type of argument is integer
                    default = 224) # Default is 224
    # Epochs
    ap.add_argument('-e',
                    '--epochs',
                    help = 'number of epochs the model should run',
                    type = int, # Type of argument is integer
                    default=25) # Default is 25
    # Batches
    ap.add_argument('-b',
                    '--batches',
                    help = 'Batch size to group dataset into - please use batch size with powers of 2 (8, 16, 32 etc.) as these are compatible with computer memory',
                    type = int, # Type of argument is integer
                    default=32) # Default is 32
    
    args = ap.parse_args()
 
    # cnn_artists is imported
    cnn_artists_class = cnn_artists(args)
    
    trainX, trainY = cnn_artists_class.data_preparation(args.train, # Train and test data is defined 
                                                                   save_labels = True)
    # Test
    testX, testY = cnn_artists_class.data_preparation(args.validation,  # Train and test data is defined 
                                                                   save_labels = False)
   # Model is defined and compiled
    cnn_artists_class.model_preparation(trainX, trainY, testX, testY)

    # Model is evaluated
    cnn_artists_class.model_evaluation(testX, testY)

    # Learning and accuracy curves are saved in a plot
    cnn_artists_class.plot_history()

        
        
if __name__=="__main__":
    main() 
    
    
    