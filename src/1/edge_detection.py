''' Assignment description
The purpose of this assignment is to use computer vision to extract specific features from images. The script can both detect edges on the original image We_Hold_These_Truths_at_Jefferson_Memorial used in the assignment as well as other image datasets. In the script, you can use the original image or the data set from Kaggle containing car license plates (can be found here https://www.kaggle.com/mobassir/fifty-states-car-license-plates-dataset).
'''

# Packages 
import os # Path operations
import sys # Path operations

import cv2 # Image operations
import numpy as np # Numeric operatoins

from pathlib import Path # Used to search in folder
import argparse # To make arguments from the terminal


class canny_calculations: # Defining class called 'canny_calculations'      
    ''' This is where the class is defined
    '''

    def __init__(self, args): # Always starting with "self"
        '''
        In init args are defined in order to make them belong to the class and use them throughout the functions in this class
        '''
        # Assigning args to the class with the 'self' attribute
        self.args = args 

    def preprocess(self):
        '''
        This function performs preprocessing actions on a picture. It first crops the image to a desired size and afterwards converts color scale to greyscale and applies a Guassian filter w 5x5 kernel
        '''
        # Now I'll draw a green, rectangular box to make a region of interest (ROI) - since we will be detecting letters I'll crop the area around these
        # I'll start by inspecting the dimensions of the picture and taking the middle area
        # Different cropping applies to the different conditions
        if self.args['condition'] == 'Memorial':
            start_x = int(self.image.shape[1]/4)
            end_x = int(3*(self.image.shape[1]/4))
            start_y = int(self.image.shape[0]/4)
            end_y = int(4*(self.image.shape[0]/4))
        
        # Different cropping applies to the different conditions
        elif self.args['condition'] == 'Cars':
            start_x = int(0.5*self.image.shape[1]/6)
            end_x = int(5.5*(self.image.shape[1]/6))
            start_y = int(self.image.shape[0]/6)
            end_y = int(5.5*(self.image.shape[0]/6))
      
        # Assigning the cropped image to a self object to use it throughout the script without having to use return to obtain it   
        self.cropped = self.image[start_y:end_y, start_x:end_x]
        
        
        # Now I'll see whether these coordinates gets an amount of text that I can analyze
        self.image_ROI = cv2.rectangle(self.image.copy(), # Copy to avoid deconstructing the original image
                      (start_x, start_y), # Start on x and y axis
                      (end_x, end_y), # End on x and y
                      (0, 255, 0), # Color - in this case green
                      2) # Thickness of 2

        # Now I'll see how we can extract edges with canny and find the letters. I'm starting by turning the picture into greyscale
        self.grey_image = cv2.cvtColor(self.cropped, cv2.COLOR_BGR2GRAY)
        self.blurred = cv2.GaussianBlur(self.grey_image, # Greyscale
                                        (5,5), # image, 5x5 kernel
                                        0) # 0 = amount of variation from the mean you can take into account - high = high variation
    
    def auto_canny(self):
        '''
        This function performs canny edge detection. It first saves the 5% and 95% percentile of the image values and uses these as threshold when detecting edges on the preprocessed picture (blurred w Gaussian filter)
        '''
        
        # Defining the canny edge detection
        P5 = np.percentile(self.grey_image, 5)  
        P95 = np.percentile(self.grey_image, 95)
        
        # With histogram threshold
        canny = cv2.Canny(self.blurred, P5, P95) # Min and max threshold - find parameters automatically or manually
        
        # Defining the contours in the image
        (contours,_) = cv2.findContours(canny.copy(), # using np function to make a copy rather than destroying the image itself
                         cv2.RETR_EXTERNAL, 
                         cv2.CHAIN_APPROX_SIMPLE) 
       
        self.cropped_contours = cv2.drawContours(
                         self.cropped.copy(), # image, contours, fill, color, thickness
                         contours,
                         -1, # whihch contours to draw. -1 will draw contour for every contour that it finds
                         (0,255,0), # contour color
                         2)

    def write_img(self, filename):
        '''
        This function takes no input but loops over the previously produced cropped image, the image with ROI and the image with highlighted text after canny edge detection. 
        '''
        # List with image versions
        versions = [self.cropped, self.image_ROI, self.cropped_contours] 
        
        # Names of different versions
        names = ['cropped.jpg','with_ROI.jpg','letters.jpg']
        
        # Name of output folder defined in commandline arguments
        output = self.args['output_folder'] 
        
        # Zipping versions and names to loop over both objects
        for (version,name) in zip(versions,names):             
            pic_name = os.path.join(output, f'{os.path.splitext(os.path.basename(filename))[0]}_{name}') # Defining the name as the base name w/o path or extension - e.g. 'Arizona' 
            
            # Saving the version and the outputname
            cv2.imwrite(pic_name, version) 

    def loop(self):
        '''
        Function that loops over all pictures in a folder and performs above-defined actions on them
        '''
        if self.args['condition'] == 'Memorial':
            # Takes every file with .JPG extension 
            for filename in Path(self.args['data_folder']).glob("*.JPG"): 
                
                # Str because thats what cv2.imread like
                self.image = cv2.imread(str(filename)) 
                
                # Preprocess ie. crop, greyscale and blur
                self.preprocess() 
                
                # Canny edge detection
                self.auto_canny() 
                
                # Save all images
                self.write_img(filename=filename) 
        elif self.args['condition'] == 'Cars':
             # Takes every file with .jpg extension 
            for filename in Path(os.path.join(self.args['data_folder'], 'fifty-states-car-license-plates-dataset')).glob("*.jpg"):
                
                 # Str because thats what cv2.imread like
                self.image = cv2.imread(str(filename))
                
                # Preprocess ie. crop, greyscale and blur
                self.preprocess() 
                
                # Canny edge detection
                self.auto_canny() 
                
                # Save all images
                self.write_img(filename=filename) 
        
    
def main(): # Now I'm defining the main function where I try to make it possible executing arguments from the terminal
    # add description
    ap = argparse.ArgumentParser(description = "[INFO] creating canny edge detection") # Defining an argument parse

    ap.add_argument("-d", 
                    "--data_folder",  # Argument 1
                    required=False, # Not required
                    type = str, # The input type should be a string
                    default = os.path.join("..","..","data","1"), # Default - this is where my data is
                    help = "str of data_folder, default is ../../data/1") # Help function
    
    ap.add_argument("-o", 
                    "--output_folder",  # Argument 2
                     required=False, # Not required
                     type = str, # The input type should be a string
                     default = os.path.join("..","..","out","1"), # Default - where to put output data
                     help = "str of output folder, default is ../../out/1") # Help function
    
    
    ap.add_argument("-c",  # Argument 3
                    "--condition", 
                    required=True, # Required
                    choices=['Memorial', 'Cars'], # Possibilities to choose
                    type = str, # The input type should be a string
                    help = "REQUIRED. Condition - 'Memorial' for single image of Jefferson Memorial or 'Cars' for multiple car license plate pictures") # Help function
    
    # Adding them together
    args = vars(ap.parse_args()) 
    
    # Defining what they corresponds to in the canny class 
    image_operator = canny_calculations(args) 
    
     # Running loop function
    image_operator.loop()
   
    
if __name__ == "__main__":
    main()
    
    
    
    