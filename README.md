# Disaster Response Pipeline Project


## Installation
The application runs on python in a Flask framework and a MySQL database. The following installation is needed

- NLTK
- Flask
- Plotly
- Joblib
- Sqlalchemy
- Sklearn

## Working method
The application consists of three modules. The data processer, the classifier trainer and the web app.

### The data processor
The data for the classifier is offered through two csv files received from a stakeholder. This module reads the csv files, merges them
and cleans the data. Following that the data is stored in a database table which consists of messages and categories for those messages.

### The classifier trainer
The classifier uses natural language processing to fit a prediction model on the datatabel. This model is stored such that it can be used for the prediction. 

### The web app
The web app offers an overview of the media genres used to communicate needs for aid. Based on the trainingset in the database and the model a prediction can be made. For instance when writing "Flooded need water" the results will show categories like request, aid related and medical help.


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Be aware:
For the database a MySQL database is used, running on a SQLalchemy engine.
