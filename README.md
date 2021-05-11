## Visual Analytics exam portfolio in Cultural Data Science on Aarhus University, 2021
### Marie Damsgaard Mortensen

## Content

This portfolio consists of 4 projects - 3 class assignments made throughout the semester (number 1-3) and 1 self-assigned (number 4). 

<details open>
<summary> Assignment descriptions </summary>
<br>
    
| Assignment | Description|
|--------|:-----------|
| 1 | Canny edge detection using openCV |
| 2 | Classifying digits with Neural Network and multinomial Logistic Regression |
| 3 | Impressionist artist classification with fully-connected and pre-trained Neural Network |
| 4 | Detecting architectural patterns with EfficientNet |
    
</details>


## Structure

To familiarize yourself with the structure of the repository, please see the table below that describes the main folders: 

| Folder | Description|
|--------|:-----------|
| data | contains data used in the analyses - inside this folder there it is specified how data for each assignment can be accessed |
| src | source code and main scripts |
| out | results (e.g. learning curves, classification reports, plots) |
| utils | utility functions used in the scripts |


## 🔧 Setup

To see and run the code with the correct packages installed, please clone the GitHub repository to a place on your computer where you'd like to have it by typing:

```
git clone https://github.com/marmor97/cds-visual-exam # Clone repository to local machine or server

cd cds-visual-exam # Change directory to the repository folder

bash create_vision_venv.sh # Creates a virtual environment
```

Every time you wish to run any of the scripts, please type the following commands:

```
source cv101_marie/bin/activate # Activates virtual environment
```

Now you can move to any script in the src folder and execute it:

```
# Running script in assignment 1 as an example - exchange with whatever script you wish to run

cd src/1/ # Changing to src and assignment folder 

python3 edge_detection.py # Run script
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## ❓ Questions and contact  
To ask questions, please contact me on mariemortensen97@gmail.com.
