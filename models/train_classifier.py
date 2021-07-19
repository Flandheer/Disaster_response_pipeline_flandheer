import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine
sys.path.append('../data')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import sqlite3
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

import sys
import pickle

def load_data(database_filepath):
    '''
    Function to load data from the ETL process. Also creates the label and features used for training
    the model
    :param database_filepath: reference to the database
    :return:
    X: Label
    y: Features
    '''
    # Create connection with
    db_connect = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("Messages", con = db_connect)


    X = df['message']

    Y = df[['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']]

    return X,Y


def tokenize(text):
    '''
    Function to demarcate sections of a string. The resulting tokens are used for building the model
    :param text: Input text from end users
    :return:cleaned tokens usable for analysis
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []

    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens


def build_model():
    '''
    Function to build a model for classifying and predicting needs of end user based on text input. Uses a
    machine learning pipeline to process information.

    :return: A GridSearchCV that can be used to fit training data and build a model
    '''

    # set methods and training flow
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(n_estimators = 1))),
    ])

    #set parameters
    parameters = {
        'vect__max_df': (0.5, 1.0),
        'clf__estimator__n_estimators': [5, 10]
    }

    #Perform gridSearch to get best parameters for task
    cv = GridSearchCV(pipeline, param_grid=parameters, cv = 5)

    return cv


def evaluate_model(model, X_test, Y_test):
    '''
    Function to print an evaluation report (precision, recall, F1-score) for every category in the data.
    '''

    Y_prediction = model.predict(X_test)

    for index, value in enumerate(Y_test):
        print('Report of the f1 score, precision and recall for category: {} \n'.format(value))
        print(classification_report(Y_test.iloc[:, index], Y_prediction[:, index]))
        print()

    pass


def save_model(model, model_filepath):
    '''
    Function to save the model as .pkl file such that it can be used without training the model.

    :param model: Model created with the machine learning pipeline and training data
    :param model_filepath: Location to save the .pkl file
    :return: .pkl file
    '''

    filename = model_filepath
    pickle.dump(model, open(filename,'wb'))

    pass


def main():
    '''
    Main code to run classifier training module. To run the following arguments should be given
        DisasterResponse.db ../models
    Add tot parameters in your IDE or in your command line following
        python3 train_classifier.py
    :return: Machine learning model that can be used to categorize text messages
    '''

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()