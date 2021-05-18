'''
This project attempts to classify 25 different architectural styles using the pretrained resnet50 model combined with final layers defined by me. The dataset was made by https://www.kaggle.com/wwymak/architecture-dataset. It features 25 different architectural styles with in total 10113 images. Additionally, the assignment attempts to produce feature heatmaps of a sample from each style. Lastly, it tries to make it possible to upload ones own images and see which style has the strongest association with it.
'''

# Plotting 
import matplotlib.pyplot as plt
import seaborn as sns 

# Path operations
import sys,os,pathlib

# To pick random picture from path
import random 

# Saving model summaries
from contextlib import redirect_stdout 

import numpy as np

# Tensorflow and keras tools
import tensorflow as tf
import keras

import keras.preprocessing
import tensorflow_datasets as tfds
from tensorflow.keras import layers 
from tensorflow.keras import Model 
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)

from tensorflow.keras.utils import plot_model 
from tensorflow.keras import backend as K

# Optimizers
from tensorflow.keras.optimizers import SGD, Adam 

# Image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array)
import cv2

# Evaluation modules
from sklearn.metrics import (classification_report, confusion_matrix)


# Commandline arguments
import argparse


class architectural_resnet50:
    '''
    Class object using a pre-trained ResNet50 network to classify architecture, predict unseen pictures and save heatmaps of important features in each class.
    '''
    def __init__(self, args):
        self.args = args # Assigns args to use it throughout the script

    
    def get_data(self):
        '''
        Loads data into two splits.
        '''
        print("[INFO] Loading data and splitting into train and validation splits...")
        
        training_data = tf.keras.preprocessing.image_dataset_from_directory(
        # The function image_dataset_from_directory needs a directory of where the data is located. When it has this, it can 'infer' the labels from the subdirectories, because each of the subdirectories contain the name of the class they include.
        self.args['path'],       
        #'categorical' means that the labels are encoded as a categorical vector - one-hot encoding
        label_mode = 'categorical', 
        # Set seeds
        seed=2021,
        # Alphanumeric order.
        shuffle = False, 
        # 80/20 split
        validation_split = 0.20,
        # What this set is
        subset = "training",
        # Image size
        image_size=(224, 224))
        
        # Getting labels
        class_names = training_data.class_names
        
        # Normalize
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255) # Defining normalization layer       
        training_data = training_data.map(lambda x, y: (normalization_layer(x), y)) # Applying the above defined layer

        # Validation                                                              
        validation_data = tf.keras.preprocessing.image_dataset_from_directory( 
        self.args['path'], 
        label_mode = 'categorical', 
        seed=2021,
        shuffle = False, 
        validation_split = 0.20,
        subset = "validation",
        image_size=(224, 224))
        
        # Normalize
        validation_data = validation_data.map(lambda x, y: (normalization_layer(x), y)) # Applying the above defined layer
        
        return training_data, validation_data, class_names
               
        
    def def_model(self, n_classes=25):
        '''
        Function that defines the ResNet50 model and its final layers.
        '''
        
        print("[INFO] Defining model...")

        # Removing any saved state of the model
        np.random.seed(1)
        random.seed(2)
        if tf.__version__[0] == '2':
            tf.random.set_seed(3)
        else:
            tf.set_random_seed(3)
        print("RANDOM SEEDS RESET")
 
        # del model
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

        # Importing resnet50 with keras.applications 
        resnet50 = tf.keras.applications.ResNet50( 
            input_shape=(224, 224, 3), # Resnet is trained on 224x224
            pooling = 'avg', # Average pooling
            classes = n_classes, # 25
            include_top=False) # False because I will define my own fully-connected

        for layer in resnet50.layers: # Running through every layer of the model 
            layer.trainable = False # Setting the option to train the layers again to false

        # Add new classifier layers
        flat1 = Flatten()(resnet50.layers[-1].output) # The return value of Flatten() is a function - Reverse index - very confusing syntax
        class1 = Dense(256, 
                       activation='relu')(flat1)

        output = Dense(n_classes, 
                       activation='softmax')(class1)

        # Define new model
        self.resnet50 = Model(inputs=resnet50.inputs, # inputs 
                      outputs=output) # outputs
        
        # Summarize and save
        with open(os.path.join(self.args['outpath'], 'modelsummary.txt'), 'w') as f:
            with redirect_stdout(f):
                self.resnet50.summary()   

        # Compiling model
        # We compile using ADAM and categorical cross-entropy as the loss function.
        self.resnet50.compile(optimizer=Adam(lr=0.0001),
                      loss='categorical_crossentropy', # Multiclass classification
                      metrics=['accuracy'])  

    def fit_model(self, training_data, validation_data, class_names):
        '''
        Fitting ResNet50 to training data and gathering predictions.
        '''
        print("[INFO] Fitting model...")
        # Fitting model
        self.H = self.resnet50.fit(
              training_data,
              validation_data=(validation_data), # Data on which to evaluate the loss at the end of each epoch - model will not be trained on this data. 
              batch_size=128,
              epochs=self.args['epochs'])

        # Predictions on validation set
        y_val = np.concatenate([y for x, y in validation_data], axis=0)
        x_val = np.concatenate([x for x, y in validation_data], axis=0)
        predictions_all = self.resnet50.predict(x_val)
        
        report = classification_report(y_val.argmax(axis=1), # y true - .argmax(axis=1) to retrieve actual labels
                     predictions_all.argmax(axis=1), # y pred .argmax(axis=1) to retrieve actual labels
                     target_names=class_names, # labels defined in the first function
                     output_dict=True)

        
    def plot_history(self):
        '''
        Creates plot of training and validation accuacy and loss.
        '''
        print("[INFO] Creating plot of learning curves...")
        if self.args['train_model'] == 'Yes':
            # Visualize performance
            plt.style.use("fivethirtyeight")
            plt.figure()
            plt.plot(np.arange(0, self.args['epochs']), self.H.history["loss"], label="train_loss")
            plt.plot(np.arange(0, self.args['epochs']), self.H.history["val_loss"], label="val_loss")
            plt.plot(np.arange(0, self.args['epochs']), self.H.history["accuracy"], label="train_acc")
            plt.plot(np.arange(0, self.args['epochs']), self.H.history["val_accuracy"], label="val_acc")
            plt.title("Training Loss and Accuracy")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.tight_layout()

            # Specifying save location and name
            plt.savefig(os.path.join(os.path.join(self.args['outpath'], "resnet50_learning_curves.jpg"))) 
            plt.show()    

    def predict_unseen(self, class_names):
        '''
        Function that takes pretrained and fitted resnet50 model and predicts new image. Returns a print of predictions and saves a barplot of these.
        '''
        print("[INFO] Predicting unseen image...")
        
        # Predict unseen image
        image = tf.keras.preprocessing.image.load_img(self.args['unseen_image_path'],
                                                      target_size = (224, 224)) # 224x224 pixels

        # Preprocessing 
        x = keras.preprocessing.image.img_to_array(image)/255. # Scaling picture
        x = np.expand_dims(x, axis=0) # Adding an extra dimension - making it a 4-d tensor

        predictions = self.resnet50.predict(x)

        # Convert the probabilities to class labels
        preds_w_names = list(zip(class_names, predictions[0]))

        # Print the names and predictions
        print(preds_w_names) 

        # Choosing a palette from Color Brewer palette
        palette = sns.color_palette("Paired")

        # This creates a figure 8 inch wide, 4 inch high
        plt.figure(figsize=(8,4)) 

        # Defining barplot as label_names on x-axis and prediction values on y-axis with the palette specified above
        bplot = sns.barplot(label_names, predictions[0], palette = palette)

        # Rotate x-labs for better visibility
        bplot.set_xticklabels(bplot.get_xticklabels(), rotation=40, ha="right")

        plt.tight_layout()

        # Specifying save location and name
        plt.savefig(os.path.join(os.path.join(self.args['outpath'], "new_pic_probs.jpg"))) 

        # Show plot
        plt.show()

    def grad_cam(self, last_layer = 'conv5_block3_out'): 
        
        '''
        Function that uses the last convolutional layer of a model to perform grad-cam and make heatmaps.
        '''
        print("[INFO] Performing GradientTape")

        image = tf.keras.preprocessing.image.load_img(self.random_pic_path, target_size = (224, 224))

        # Preprocessing 
        x = keras.preprocessing.image.img_to_array(image)/255.
        x = np.expand_dims(x, axis=0)

        predictions = resnet50.predict(x)

        with tf.GradientTape() as tape:

            # make sure the name here corresponds to the final conv layer in your network
            last_conv_layer = self.resnet50.get_layer(last_layer) # Last convolutional layer

            # First, we create a model that maps the input image to the activations
            # of the last conv layer as well as the output predictions    
            iterate = tf.keras.models.Model([self.resnet50.inputs], 
                                            [self.resnet50.output, last_conv_layer.output])

            # Then, we compute the gradient of the top predicted class for our input image
            # with respect to the activations of the last conv layer
            model_out, last_conv_layer = iterate(x) # Collecting the gradients of the model
            class_out = model_out[:, np.argmax(model_out[0])]

            # This is the gradient of the output neuron of the last conv layer
            grads = tape.gradient(class_out, 
                                  last_conv_layer)

            # Vector of mean intensity of the gradient over a specific feature map channel
            pooled_grads = K.mean(grads, axis=(0, 1, 2)) # Just like in numpy, you can define the axis along you want to perform a certain operation

        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
        heatmap = np.maximum(heatmap, 0) 
        heatmap /= np.max(heatmap) # Equivalent to heatmap = heatmap / np.max(heatmap)
        heatmap = heatmap.reshape((7, 7)) # When using the last convolutional layer of the resnet50 model, it needs the sizes to be 7x7 pixels

        # Load the original image
        original = cv2.imread(str(self.random_pic_path))

        # Heatmap should be semi transparent
        intensity = 0.5

        # Resize the heatmap to be the original dimensions of the input 
        heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))

        heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

        # Multiply heatmap by intensity and 'add' this on top of the original image
        superimposed = (heatmap * intensity) + original

        return superimposed


# Grad CAM - fancy way of saying that we are taking advantage of the way the model learns gradients - uses gradients from network to figure out how to modify
    def heatmap_samples(self):
        '''
        Function that goes takes sample images and performs grad_cam() to get heatmaps. It saves these in the defined output path.
        '''
        print("[INFO] Taking samples from classes and making heatmap...")
        # list all files in dir
        heatmaps = []
        styles = os.listdir(self.args['path'])
         
        # Go through every subfolder of "architectural-styles-dataset"
        for subfolder in sorted(styles):
            
            # Getting list of all pictures in the directories
            pics = os.listdir(os.path.join(f"{self.args['path']}",f"{subfolder}"))
            
            # Choose random picture
            random_pic = random.choice(pics) 

            # Save entire path
            self.random_pic_path = os.path.join(f"{path}",f"{subfolder}", f"{random_pic}")

            # Using the above-defined grad_cam function to get a superimposed version of heatmap of the image
            superimposed = self.grad_cam(self.resnet50, self.random_pic_path)

            # Save to lists with heatmaps
            heatmaps.append(superimposed)

            # Save the heatmap as well
            cv2.imwrite(os.path.join("heatmaps", f"super_im_{subfolder.replace(' ', '_')}.jpg"), superimposed)    
            
            print(f"[INFO] Super imposed image saved as super_im_{subfolder.replace(' ', '_')}.jpg")
        
        return heatmaps
   
    def style_heatmaps(self, class_names, heatmaps):
        '''
        Takes heatmap pictures of all classes and and creates a plot with a grid of these pictures.
        '''
        # Plotting all
        import numpy as np
        import matplotlib.pyplot as plt

        w=224
        h=224
        fig=plt.figure(figsize=(14, 10))
        columns = 5
        rows = 5

        # Loop running 25 times
        for i in range(1, columns*rows +1): 
            # Take the i-1th image in the list heatmap
            img = heatmaps[i-1] 
            # Creating subplot
            ax = fig.add_subplot(rows, columns, i) 
            ax.set_title(class_names[i-1])
            # Changing pixelvalues to RGB instead of BGR
            plt.imshow(img.astype(int)[:,:,::-1]) 
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.join(outpath, "style_heatmaps_plot.jpg"))) 
        plt.show()

        
def main():
    
    ap = argparse.ArgumentParser(description = "[INFO] creating resnet50 model") # Defining an argument parse
    
    ap.add_argument("--outpath", "-o", 
                    help = "Path where plots, figures and heatmaps are saved",
                    type = str,
                    default = os.path.join("..","..","out","4"))
    
    ap.add_argument("--path", "-p", 
                    default = os.path.join("..","..","data","4","architectural-styles-dataset"),
                    type = str,
                    help = "path to folder containing train and validation images")

    ap.add_argument("--epochs", "-e",
                    default = 25, 
                    type = int, 
                    help = "Number of epochs")

    ap.add_argument("--unseen_image_path", "-u",
                    default = os.path.join("..","..","data","4","unseen_img.jpg"),
                    type = str,
                    help = "Path to unseen image that should be predicted")
    
    args = vars(ap.parse_args()) # Adding them together

    archi_resnet50 = architectural_resnet50(args)
    
    training_data, validation_data, class_names = archi_resnet50.get_data()
    
    archi_resnet50.def_model()
    
    archi_resnet50.fit_model(training_data, validation_data, class_names)
   
    archi_resnet50.plot_history()
    
    archi_resnet50.predict_unseen(class_names)
    
    heatmaps = archi_resnet50.heatmap_samples()
        
    archi_resnet50.style_heatmaps(heatmaps, class_names)
    
if __name__ == '__main__':
    main()