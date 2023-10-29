
# ZZSC5836 Data Mining and Machine Learning - Assignment 2

## Data processing and linear regression

Abalone Dataset: Predict the Ring age in years

> Predicting the age of abalone from physical measurements. The age of abalone is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope -- a boring and time-consuming task. Other measurements, which are easier to obtain, are used to predict the age. Further information, such as weather patterns and location (hence food availability) may be required to solve the problem.  

> Source: https://archive.ics.uci.edu/ml/datasets/abalone

>Sex / nominal / -- / M, F, and I (infant)
Length / continuous / mm / Longest shell measurement
Diameter / continuous / mm / perpendicular to length
Height / continuous / mm / with meat in shell
Whole weight / continuous / grams / whole abalone
Shucked weight / continuous / grams / weight of meat
Viscera weight / continuous / grams / gut weight (after bleeding)
Shell weight / continuous / grams / after being dried
Rings / integer / -- / +1.5 gives the age in years (ring-age)

> Ignore the +1.5 in ring-age and use the raw data


### Overview

Create a report and include the Data Processing and Modelling tasks given below. 

#### Data processing (10 Marks):

1. Clean the data (eg. convert M, F and I to 0, 1 and 2). You can do this with code or simple find and replace (2 Marks).
2. Develop a correlation map using a heatmap and discuss major observations (2 Marks).
3. Pick two of the most correlated features (negative or positive) and create a scatter plot with ring-age. Discuss major observations (2 Marks).
4. Create histograms of the two most correlated features, and the ring-age. What are the major observations?  (2 Marks)
5. Create a 60/40 train/test split - which takes a random seed based on the experiment number to create a new dataset for every experiment (2 Marks).
6. Add any other visualisation of the dataset you find appropriate (OPTIONAL). 

#### Modelling  (10 Marks):

1. Develop a linear regression model using all features for ring-age using 60 percent of data picked randomly for training and remaining for testing. Visualise your model prediction using appropriate plots. Report the RMSE and R-squared score. (4 Marks)
2. Develop a linear regression model with all input features
   -  without normalising input data
   -  with normalising input data. (2 Marks)
3. Develop a linear regression model with two selected input features from the data processing step. (2 Marks)
4. In each of the above investigations, run 30 experiments each and report the mean and std of the RMSE and R-squared score of the train and test datasets. Write a paragraph to compare your results of the different approaches taken. Note that if your code can't work for 30 experiments, only 1 experiment run is fine. You won't be penalised if you just do 1 experiment run. (2 Marks)

#### Submission

1. Upload your code in this workspace provided in Ed. Ensure that the code is running in order to get full marks.
2. Upload a pdf of the report to Moodle as well as in this workspace in Ed with the code.  

The code should use relevant functions and create required outputs. You can also use python notebooks as needed. 

#### Instructions

1. You can use the existing code given in the course or any other sources online. You need to cite the code as a reference in the report.
2. You can use Python or R, or both depending on whatever is suitable.
3. You can use Ed directly to write code or use your computer but need to ensure that it runs on Ed. If it does not run on Ed, you can upload a screenshot of it running on your computer.  
