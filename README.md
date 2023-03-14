# Restaurant Data Analysis

## Summary | Problem Statement

### Summary :
    Abstract:
        The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit
    
    Data Set Information:
        The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. 
    
    Analysis Steps:
        Atribute information Analysis.
        Deep Learning (ANN)
    
    Source:
        Dataset from : http://archive.ics.uci.edu/ml/datasets/Bank+Marketing#    . 

### About Dataset :
 * [Bank Marketing](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)
    - Data Cleaning:
        - Deleting redundant columns.
        - Renaming the columns.
        - Dropping duplicates.
        - Cleaning individual columns.
        - Check for some more Transformations
        - Handling outliers
        - Scaling
        - Handling imbalanced data
        - Hyperparameter tuning
    
    - Data Columns:
        - Age (numeric)
        - Job : type of job (categorical: 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown')
        - Marital : marital status (categorical: 'divorced', 'married', 'single', 'unknown' ; note: 'divorced' means divorced or widowed)
        - Education (categorical: 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown')
        - Default: has credit in default? (categorical: 'no', 'yes', 'unknown')
        - Housing: has housing loan? (categorical: 'no', 'yes', 'unknown')
        - Loan: has personal loan? (categorical: 'no', 'yes', 'unknown')
        - Contact: contact communication type (categorical:
'cellular','telephone')
        - Month: last contact month of year (categorical: 'jan', 'feb', 'mar',
…, 'nov', 'dec')
        - Day_of_week: last contact day of the week (categorical:
'mon','tue','wed','thu','fri')
        - Duration: last contact duration, in seconds (numeric). Important
        - note: this attribute highly affects the output target (e.g., if
duration=0 then y='no'). Yet, the duration is not known before a call
is performed. Also, after the end of the call y is obviously known.
Thus, this input should only be included for benchmark purposes and
should be discarded if the intention is to have a realistic
predictive model.
        - Campaign: number of contacts performed during this campaign and for
this client (numeric, includes last contact)
        - Pdays: number of days that passed by after the client was last
contacted from a previous campaign (numeric; 999 means client was not
previously contacted)
        - Previous: number of contacts performed before this campaign and for
this client (numeric)
        - Poutcome: outcome of the previous marketing campaign (categorical:
'failure','nonexistent','success')  
        - Emp.var.rate: employment variation rate - quarterly indicator
(numeric)
        - Cons.price.idx: consumer price index - monthly indicator (numeric)
        - Cons.conf.idx: consumer confidence index - monthly indicator
(numeric)
        - Euribor3m: euribor 3 month rate - daily indicator (numeric)
        - Nr.employed: number of employees - quarterly indicator (numeric)

### Problem Statement :  
problem is to predict whether a client will subscribe to a term deposit or not, given a set of features about the client and the marketing campaign. This is a binary classification problem, where the target variable is a binary variable indicating whether the client has subscribed to the term deposit or not. The features may include demographic information about the client, such as age, gender, education level, and occupation, as well as information about the marketing campaign, such as the type of communication used, the number of contacts made, and the time of day the contact was made. The goal of the machine learning model is to learn patterns in the data that can be used to predict whether a client is likely to subscribe to the term deposit or not, based on these features.

## Table of Contents
  - [Requirements Packages](#requirements-packages)
  - [Basic Analysis](#basic-analysis)
  - [Feature Engineering](#feature-engineering)
  - [Model Selection](#model-selection-and-evaluations)
  - [Model Selection and Evaluations](#model-selection-and-evaluations)
  

## Requirements Packages
![foxdemo](https://github.com/SarkarPriyanshu/Bank_Marketing/blob/main/Img/requirements.png?raw=true)

## Basic Analysis
### Analysis of Data
 
>>
   * Asking basic questions
        - What is data dimentionality ?
        - How data looks?
        - What data type we are dealing with?
        - Is there any null values to handle?
        - Is there any duplicated rows or columns?
        - How data looks in terms of math?
        - Is there any correlation in independent and dependent columns?

    * Observations:
        - Skewed data , feature Tranformation
        - Outliers handling
        - Handling high cardinality
        - Scaling
        - Imbalanced Data
        - Multicolinearity

![foxdemo](https://github.com/SarkarPriyanshu/Bank_Marketing/blob/main/Img/multicolinearitypng.png?raw=true)




## Feature Engineering 
###  Handling Multi-colinearity:
>       
        - Observations:
            - 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'nr.employed' are the feature that are causing muilticolinarity 
        
        
[Ways to handle Multi-colinarity](https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/)
>
        - handle Multicolinearity
            - Removing all four columns from dataset, euribor3m have same weightage as of four columns
            
![foxdemo](https://github.com/SarkarPriyanshu/Bank_Marketing/blob/main/Img/multicolinearity2png.png?raw=true)        
           

### Cardinality in dataset
>>
    * Handling Categorical Columns
        - Observation: 
            - After handling multicolinearity we deal with cardinality.
            - In this dataset we have around 10 categorical columns
            - Instead of creading multiple columns which lead to high dimentionality and increases time while traning, what we do here is look for monotonic relation between independent categorical columns and target column
            - Advantage of this technique less dimentionality and increase in traning time for ANN model
        
![foxdemo](https://github.com/SarkarPriyanshu/Bank_Marketing/blob/main/Img/cardinalitypng.png?raw=true)

![foxdemo](https://github.com/SarkarPriyanshu/Bank_Marketing/blob/main/Img/cardinalityfig1png.png?raw=true)
    
    * Handling Categorical Custom class for pipeline
            - We created a custom class which can work with sklearns pipline.
![foxdemo](https://github.com/SarkarPriyanshu/Bank_Marketing/blob/main/Img/customclass.png?raw=true)            
  
### Outliers in dataset  
    
    * handling Outliers
        - Observations: 
            - 'age','duration','campaign' are the columns which need to be handling in terms of outlier
            - We are going to use capping IQR technique to handle outliers


## Feature Engine Pipeline
[Feature Engine](https://feature-engine.readthedocs.io/)
![foxdemo](https://github.com/SarkarPriyanshu/Bank_Marketing/blob/main/Img/pipeline.png?raw=true)
       
    

## Model Selection and Evaluations
> ### Threshold Discrimination
[Yellowbrick](https://www.scikit-yb.org/en/latest/)

>>
     We are going to use logistic regression to figure out threshold for classification of this binary class classification:
    
![foxdemo](https://github.com/SarkarPriyanshu/Bank_Marketing/blob/main/Img/discriminaltionthreshold.png?raw=true)
    
     Observation: 
        0.2 - 0.4 is the best threshold to classify this binary class classification problem.
![foxdemo](https://github.com/SarkarPriyanshu/Bank_Marketing/blob/main/Img/Thresholdcompare.png?raw=true    )        

        
> ### ANN 

> ##### Case 1 (Base Model with random layers and neurons)

![foxdemo](https://github.com/SarkarPriyanshu/Bank_Marketing/blob/main/Img/BaseAnnModel.png?raw=true)
    
    Observations:
        - Overfitting 
![foxdemo](https://github.com/SarkarPriyanshu/Bank_Marketing/blob/main/Img/basemodelannaccuracyandloss.png?raw=true)

    What are the measure we can take to reduce overfitting in ANN:
        - Early stoping
        - Hyperparameter tuning (layer,neurons,loss,activition function,optimizers)
        - Dropout 
        - Weight initialisation Methods 
        - Regularisation Methods

  

> ##### Case 2 (Base Model with less epochs)
![foxdemo](https://github.com/SarkarPriyanshu/Bank_Marketing/blob/main/Img/modelwithlessepochaccauracy.png?raw=true)
![foxdemo](https://github.com/SarkarPriyanshu/Bank_Marketing/blob/main/Img/lessthresholdandepochsmetrics.png?raw=true)
        
        Observations: 
            - As our data is imbalance we are focusing on client who are willing
            to subscribed a term deposit so we have to focus on True Positive Rate and Weighted Avg Precision of this model in terms of 'Yes' i.e nothing but the minority class.     


> ##### Case 3 (With random under sampling)
![foxdemo](https://github.com/SarkarPriyanshu/Bank_Marketing/blob/main/Img/randomundersampling.png?raw=true)

    Observations:
![foxdemo](https://github.com/SarkarPriyanshu/Bank_Marketing/blob/main/Img/randomudersamplinglossandaccuacy.png?raw=true)
![foxdemo](https://github.com/SarkarPriyanshu/Bank_Marketing/blob/main/Img/randomundersamplingmetrics.png?raw=true)
    
        Observation:
            Increase in incorrect prediction of majority class
            

> ##### Case 4 (With NearMiss)
![foxdemo](https://github.com/SarkarPriyanshu/Bank_Marketing/blob/main/Img/nearmiss.png?raw=true)

    Observations:
![foxdemo](https://github.com/SarkarPriyanshu/Bank_Marketing/blob/main/Img/nearmisslossandaccuracy.png?raw=true)
    
        Observation:
            Although it predict well for minority ('Yes') class but increase in incorrect prediction in majority ('No') class            



> ##### Case 5 (Optimising and Hyperparameter tuning ANN model)   

[KerasTuner](https://keras.io/keras_tuner/)
    We used keras tuner for hyperparameter tuning of our Model
        
        
        
        Observations:
![foxdemo](https://github.com/SarkarPriyanshu/Bank_Marketing/blob/main/Img/HyperparameterTuningAnn.png?raw=true)
![foxdemo](https://github.com/SarkarPriyanshu/Bank_Marketing/blob/main/Img/hyerparametertuninglossandaccuracy.png?raw=true)
![foxdemo](https://github.com/SarkarPriyanshu/Bank_Marketing/blob/main/Img/hyperparametertuninglayerssummary.png?raw=true)
![foxdemo](https://github.com/SarkarPriyanshu/Bank_Marketing/blob/main/Img/lessthresholdandepochsmetrics.png?raw=true)


    


> ### What else we can do to improve our model 
        - We can apply Regularisation 
        - Batch Normalisation
        - Tune learning rate
        - Weight initialisations

### [Sponsor By me]([https://github.com/SarkarPriyanshu])

## Technologies Used

- [Jupyter Notebook](https://jupyter.org/), for Cleaning, Model traning & Evaluation

## Chao

A passion project by Priyanshu Sarkar here is my [Github](https://github.com/SarkarPriyanshu) and  [CodeSandBox](https://codesandbox.io/u/SarkarPriyanshu)
