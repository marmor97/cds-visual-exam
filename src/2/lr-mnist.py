'''
Classifier benchmarks using Logistic Regression and a Neural Network (Assignment 4)

In this assignment, I will create two command-line tools which can be used to perform a simple classification task on the MNIST data and print the output to the terminal. These scripts can then be used to provide easy-to-understand benchmark scores for evaluating these models.

One python script takes the full MNIST data set, trains a Logistic Regression Classifier, and prints the evaluation metrics to the terminal.

The other python script takes the full MNIST dataset, trains a Neural Network classifier, and prints the evaluation metrics to the terminal.
'''
# Path tools
import sys, os
sys.path.append(os.path.join("..", ".."))

# Numpy and pandas for data wrangling
import numpy as np
import pandas as pd

# Import cv2 to tackle unseen images 
import cv2

# Importing clf_util from utils
import utils.classifier_utils as clf_util

# Import machine learning modules
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Importing pickle to save models
import pickle


# Command line arguments tool
import argparse


class lr_mnist:
    def __init__(self, args):
        self.mnist = fetch_openml('mnist_784', # Type of dataset
                                  version=1, 
                                  return_X_y=False) # This loads the entire set of the nnist handwritten digits data
        self.args = args
           
            
    def data_preparation(self):
        '''
        This function transform and partitions data to in preparation to modelling. 
        '''
        # Firstly, X is defined as the 'data' attribute of 'self.mnist'. Then it is turned into an array.  
        X = np.array(self.mnist.data)   
        
        # Y is defined as an array of the self.mnist 'target' attribute
        self.y = np.array(self.mnist.target) 
        
         # Here we perform a 'min-max normalization' to squeeze pictures in to smaller value space. The maximum value is transformed into a 1 and the minimum value becomes 0 after the transformation. All values in between get a number in between 0 and 1.
        X = (X - X.min())/(X.max() - X.min())
        
        # Splitting data up
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, # The input features
                                                        self.y, # The labels
                                                        random_state=9, # Random state - producing same results every time
                                                        test_size=self.args['test_split_value']) # I have selected a test / train split of 20 % / 80 %

    
    # Determining the classifier - here I'm selecting the multinomial logistic regression
    def lr_classifier(self):
        '''
        This function defines the classifier used in the script, in this case a Logistic Regression classifier. 
        '''
        self.clf = LogisticRegression(penalty = self.args['penalty'], # No penalty applied. Here you could have added either L1, L2 or elastic net as a penalization method
                             tol = self.args['tolerance'], # Tolerance for stopping criteria.
                             solver=self.args['solver'], # 'Saga' is the algorithm used in the optimization problem to help the model converge. Saga both supports multiclass problems and works well with large data sets
                             multi_class='multinomial').fit(self.X_train, self.y_train) # Class is multinumial because the classification task is not binary

    
    def evalutation(self):
        '''
        A function for evaluation of performance of the Logistic Regression - produces a print of the classification report 
        '''
        # Predictions of X_test
        y_pred = self.clf.predict(self.X_test) 
        
        # Classification report containing F-scores, accuracies + precision/recall
        report = pd.DataFrame(metrics.classification_report(self.y_test, y_pred, output_dict = True)) 
        
        # The function returns a print of the report
        print(report)
        
        report.to_csv(self.args['output_path'])
        print(f'[INFO] report is saved in {self.args["output_path"]}')
    

    def save_model(self):
        '''
        Function allowing user to save the trained model. The model is saved in the folder defined with the commandline arguments
        '''
        # Save the model
        if self.args['saving_mode'] == 'Yes':
            path = self.args['model_path']
            filename = f'{path}trained_model.sav' # Combining filename defined in commandline argument and 'trained model' with formatted strings
            pickle.dump(self.clf, open(filename, 'wb')) # Using pickle to save the model at the defined location
            

    def predict_unseen(self):
        '''
        Function that uses the logistic regression model to predict a new image
        '''
        if self.args['predict_mode'] == 'Yes':
            # Defining unique set of labels in the digit data set
            classes = sorted(set(self.y)) 
            
            # Loading the test image
            new_image_path = self.args['new_image']
            test_image = cv2.imread(new_image_path) 
            
            # Applying COLOR_BGR2GRAY to make the test image greyscale
            gray = cv2.bitwise_not(cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)) 
            
            compressed = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA) # Compressing the size of the test image into a smaller 28x28 pixel format
            clf_util.predict_unseen(compressed, self.clf, classes) # Predicting the class of the unseen image

            
def main(): 
    '''Definition of the main function''' 
    
   # I try to make it possible executing arguments from the terminal
    # Add description
    ap = argparse.ArgumentParser(description = "[INFO] creating logistic regression classifier") # Defining an argument parse
    ap.add_argument('-t', # Argument 1
                    '--test_split_value', 
                    required=False, # As I have provided a value it is not required as I have provided a default split value of 80 % / 20 %
                    type = float, # Int type
                    default = .20, # Setting default to 20 %
                    help = "Test size of dataset")
    
    ap.add_argument('-tol', # Argument 2
                    '--tolerance', 
                    required=False, # Not required
                    default = 0.1,
                    type = str, # Str type                    
                    help='Tolerance in Logistic Regression')
    
    ap.add_argument('-pen', # Argument 3
                    '--penalty', 
                    required=False, # Not required
                    choices = ['none', 'l1', 'l2', 'elasticnet'],
                    default = 'none',
                    help='Penalty in Logistic Regression - default is none')    
    
    ap.add_argument('-sol', # Argument 4
                    '--solver', 
                    required=False, # Not required
                    choices = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                    default = 'saga',
                    help='Solver in Logistic Regression - default is saga')    
    
    ap.add_argument('-s', # Argument 5
                    '--saving_mode', 
                    choices=['Yes', 'No'], 
                    required=False, # Not required
                    type = str, # Str type
                    default = 'Yes',
                    help='Save model (Yes) or dont (No)')
    
    ap.add_argument('-m', # Argument 6
                    '--model_path', 
                    required=False, # Not required
                    type = str, # Str type
                    default = os.path.join("..", "..", "out", "2"), # Setting default to the current folder
                    help = "Location where the model should be saved (path/to/saved/model)") 
    
    ap.add_argument('-p',  # Argument 7
                    '--predict_mode', 
                    choices=['Yes', 'No'], 
                    required=False, # Not required
                    type = str, # Str type                    
                    default = 'Yes',
                    help='Predict unseen photo (Yes) or dont (No)')
    
    ap.add_argument('-n', # Argument 8
                    '--new_image',
                    required=False, # Not required
                    type = str, # Str type
                    default = os.path.join("..","..","data","2","self_drawn_digit.png"), # Setting default to the current folder
                    help = "Path location of new, unseen image that the model should predict") 
    
    ap.add_argument('-o', # Argument 9
                    '--output_path', 
                    required=False, # Not required
                    type = str, # Str type
                    default = os.path.join("..","..","out","classification_report_lr.csv"), # Setting default to the current folder
                    help = "Path location of classification report") 
    args = vars(ap.parse_args()) # Adding them together
                    
    # Assigning the class and arguments to a variable                     
    lr_mnist_class = lr_mnist(args)
    
   # Using my data preparation function to make data in the right format and split data
    lr_mnist_class.data_preparation()
    
   # Here I'm defining the lr classifier
    lr = lr_mnist_class.lr_classifier()
    
    # Evaluation function to print the resulting f1 scores and accuracies
    lr_mnist_class.evalutation()
                  
    # Save model
    lr_mnist_class.save_model()
                  
    # Predict unseen image              
    lr_mnist_class.predict_unseen()
        
if __name__ == '__main__':
    main()