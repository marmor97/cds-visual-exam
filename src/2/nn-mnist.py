'''
Classifier benchmarks using Logistic Regression and a Neural Network (Assignment 4)

In this assignment, I will create two command-line tools which can be used to perform a simple classification task on the MNIST data and print the output to the terminal. These scripts can then be used to provide easy-to-understand benchmark scores for evaluating these models.

One python script takes the full MNIST data set, trains a Logistic Regression Classifier, and prints the evaluation metrics to the terminal.

The other python script takes the full MNIST dataset, trains a neural network classifier, and prints the evaluation metrics to the terminal.
'''

# Path tools
import sys,os
sys.path.append(os.path.join("..", "..")) # Appending ".." to the path in order to assess the utils folder

# Neural network with numpy
from utils.neuralnetwork import NeuralNetwork

# Importing pandas and numpy for data wrangling and 
import pandas as pd 
import numpy as np

# Import cv2 to tackle unseen images 
import cv2

# Importing clf_util from utils
import utils.classifier_utils as clf_util

# Machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import datasets

# Command line arguments tool
import argparse

class nn_mnist:
    '''
    Class object performing a not pre-trained CNN or a pre-trained VGG16 CNN on impressionist data
    '''
    def __init__(self, args):
        self.mnist = fetch_openml('mnist_784', version=1, return_X_y=False) # This loads the entire set of the nnist handwritten digits data
        self.args = args # Assigns args to use it throughout the script

    def data_preparation(self):
        '''
        This function transform and partitions data to in preparation to modelling. 
        '''
        # Firstly, X is defined as the 'data' attribute of 'self.mnist'. Then it is turned into an array and the data is turned into the required float data type. 
        X = np.array(self.mnist.data.astype("float"))    
        
        # Y is defined as an array of the self.mnist 'target' attribute
        self.y = np.array(self.mnist.target) 
        
        # Here we perform a 'min-max normalization' to squeeze pictures in to smaller value space. The maximum value is transformed into a 1 and the minimum value becomes 0 after the transformation. All values in between get a number in between 0 and 
        X = (X - X.min())/(X.max() - X.min()) 
        
        self.X_train, self.X_test, y_train, y_test = train_test_split(X, # The input features - y train and test are not transformed to self objects as they need further preprocessing
                                                        self.y, # The labels
                                                        random_state=9, # Random state - producing same results every time
                                                        test_size=self.args['test_split_value']) # Default is a test / train split of 20 % / 80 %

        
        # Convert label to binary with LabelBinarizer()
        self.y_train = LabelBinarizer().fit_transform(y_train)  
        self.y_test = LabelBinarizer().fit_transform(y_test)
        # Now they only contain 1 and 0's in a matrix

    def nn_classifier(self):
        '''
        This function defines the classifier used in the script, in this case a neural network. 
        '''
        
        # Creating a numpy array with layers for the neural network based on the hidden layer values parsed in the command-line
        layers = [] 
        
        # Input layer
        layers.append(self.X_train.shape[1]) 
        
        # Hidden layers
        [layers.append(x) for x in self.args['layer_values'].split(sep=" ")] # Len mpske

        # Output layer
        layers.append(10)
        
        layers = [int(x) for x in layers]
        
        # Numpy array
        layers = np.array(layers) 
       
                        
        # Defining the classifier with the layers above
        self.nn = NeuralNetwork(layers) 

        print(f'[INFO] {self.nn}')  # Information printed to the terminal
        
        # Fitting the above defined classifier
        self.nn.fit(self.X_train, # X
                    self.y_train, # y
                    epochs=self.args['epochs']) # N of epochs defined in the argument 'epochs'

    def evalutation(self):
        '''
        A function for evaluation of performance of NN - produces a print of the classification report 
        '''
        # Gathering predictions based on X test
        y_pred = self.nn.predict(self.X_test) 
        
        # Using .argmax(axis=1) to output labels in the right format and not probabilities
        report = pd.DataFrame(metrics.classification_report(self.y_test.argmax(axis=1), y_pred.argmax(axis=1), output_dict = True)) 
        # Printing the report to the terminal
        print(report)
        
        # Save as csv
        report.to_csv(self.args['output_path'])
        print(f'[INFO] report is saved in {self.args["output_path"]}')


    def save_model(self):
        '''
        A function that can save the NN weights as a numpy object for future use
        '''
        if self.args['saving_mode'] == 'Yes':
            np.save(self.args['model_path'], self.nn)

    def predict_unseen(self):
        '''
        Function that uses the logistic regression model to predict a new image
        '''
        if self.args['predict_mode'] == 'Yes':
            new_image_path = self.args['new_image']
            
            # Loading the test image
            test_image = cv2.imread(new_image_path) 
            
            # Applying COLOR_BGR2GRAY to make the test image greyscale
            gray = cv2.bitwise_not(cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY))
            
            # Compressing the size of the test image into a smaller pixel format
            compressed = cv2.resize(gray, (784, 33), interpolation=cv2.INTER_AREA) 
            
            # Predicting the class of the unseen image
            single_img_preds = self.nn.predict(compressed) 
            
            # Turning into class values
            predicted_class = single_img_preds.argmax(axis=1)
            
            # Counting most occuring predicted class
            count = np.bincount(predicted_class)
            print(f"I think that this is class {np.argmax(count)}")
    
def main(): # Now I'm defining the main function 
    
    # I try to make it possible executing arguments from the terminal
    # Add description
    ap = argparse.ArgumentParser(description = "[INFO] creating neural network classifier") # Defining an argument parse
    ap.add_argument("-t","--test_split_value", 
                    required=False, # As I have provided a value it is not required as I have provided a default split value of 80 % / 20 %
                    type = float, # Int type
                    default = .20, # Setting default to 20 %
                    help = "Test size of dataset")

    ap.add_argument("-l", "--layer_values", # Argument 2
                    required = False,
                    default = "32 32", 
                    type = str, 
                    help = "Hidden layers and their values in the NN classifier")        
    
    ap.add_argument("-e", 
                    "--epochs", # Argument 3
                    required = False,
                    action="store",
                    default = 20, 
                    type = int,
                    help = "Number of epochs to train the NN classifier over")
    
    ap.add_argument('-s', # Argument 4
                    '--saving_mode', 
                    choices=['Yes', 'No'], 
                    required=False, # Not required
                    default = 'Yes',
                    help='Save model (Yes) or dont (No)')
    
    ap.add_argument('-p',  # Argument 5
                    '--predict_mode', 
                    choices=['Yes', 'No'], 
                    required=False, # Not required
                    type = str, # Str type                    
                    default = 'Yes',
                    help='Predict unseen photo (Yes) or dont (No)')
    
    ap.add_argument('-n', # Argument 6
                    '--new_image',
                    required=False, # Not required
                    type = str, # Str type
                    default = os.path.join("..","..","data","2","self_drawn_digit.png"), # Setting default to the current folder
                    help = "Path location of new, unseen image that the model should predict") 
    

    ap.add_argument("-m", "--model_path", # Argument 7
                    required = False,
                    action="store",
                    default = os.path.join("..","..","out","2","nn_model.npy"), 
                    type = str,
                    help = "Path location to save trained numpy model")
    
    ap.add_argument('-o', # Argument 8
                    '--output_path', 
                    required=False, # Not required
                    type = str, # Str type
                    default = os.path.join("..","..","out","2","classification_report_nn.csv"), # Setting default to the current folder
                    help = "Path location of classification report")    
    
    args = vars(ap.parse_args()) # Adding them together

    # Assigning the class with the arguments defined above
    nn_mnist_class = nn_mnist(args)
    
   # Using my data preparation function to make data in the right format, split data and make a  matrix with dummy variables
    nn_mnist_class.data_preparation()
    
   # Here I'm defining the nn classifier. 20 epochs. 
    nn_mnist_class.nn_classifier()
    
   # nn = nn_mnist_class.nn_classifier(layer_vals = ', '.join(map(str,layer_vals)))
    nn_mnist_class.save_model()
    
    # Evaluation function to print the resulting f1 scores and accuracies
    nn_mnist_class.evalutation()
    
    # Prediction of a new, handwritten digit 
    nn_mnist_class.predict_unseen()
       
if __name__ == '__main__':
    main()
