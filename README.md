# Disaster Response Pipeline Project

## 1. Project Summary
The project is to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build an API that classifies disaster messages into categories. The project includes a web app where an emergency worker can input a new message and get classification result in several categories. The goal is to efficiently classify new messages so that relevant disaster relief agency can take actions. 
The data used in the project contains real messages that were sent during disaster events and pre-labelled categories, which can be found in the data folder. This project is part of Udacity Data Science Nanodegree.

## 2. Project Components
### 2.1. ETL Pipeline (data folder)
ETL Pipeline includes process_data.py, which is a Python script that 
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

The folder also includes ETL Pipeline Preparation.ipynb, which is a Jupyter Notebook used for prototyping ETL process.

### 2.2. ML Pipeline (models folder)
Machine Learning Pipeline includes train_classifier.py, which is a Python script that 
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

The folder also includes ML Pipeline Preparation.ipynb, which is a Jupyter Notebook was used for prototyping the Machine Learning model.

### 2.3. Flask Web App (app folder)
Flask Web app is to deploy the web app to
- display visualization of training data
- return classification results on new text using the trained model
The folder includes run.py, which is a Python script that runs the web app, and template folder with html scripts. Most of the codes for the Flask web app was provided by Udacity. 

## 3. How to run the web app
3.1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3.2. Run the following command in the app's directory to run your web app.
    `python run.py`

3.3. Go to http://0.0.0.0:3001/


## 4. Detailed information about each folder
* app
    - template
        - master.html  # main page of web app
        - go.html  # classification result page of web app
    - run.py  # Flask file that runs app

* data
    - disaster_categories.csv  # data to process
    - disaster_messages.csv  # data to process
    - process_data.py # ETL process
    - ETL Pipeline Preparation.ipynb # prototyping ETL process
    - InsertDatabaseName.db # database to save clean data to

* models
    - train_classifier.py # ML process
    - classifier.pkl  # saved model 
    - ML Pipeline Preparation.ipynb # prototyping ML process

* README.md


## 5. Remarks
The dataset used in the model is imbalanced as we can see from the below graph showing proportin of each category. 
![snapsht](https://github.com/kimjinhaha/disaster_response_pipeline_project/blob/master/proportion_categories.PNG)

While the 'related' category accounts for almost 80% of all messages, 'child_alione' categroy is almost negligible. 
The machine learning model with imbalance dataset should be evaluated carefully, since high accuracy does not necessarily mean that the model is working properly. The model might predict all instances to one category. The trained model in the project indeed shows high accuracy, but low recall for some categories. Considering the goal of the project is to categorize disaster messages correctly, it is quite important to improve recall for practical utility of the model.  


## 6. Dependencies
- Python 3.7
- requirements.txt: list of libraries used in the project.


