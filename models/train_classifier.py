import sys
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
sys.path.append('../data')
import config

import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

import sys
import pickle

def load_data(database_filepath):
    drp = database_filepath
    db_connect = create_engine(drp)
    df = pd.read_sql('disasterresponse', con = db_connect)
    X = df['message']
    Y = df[['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']]
    category_names = ['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']

    return X,Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []

    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens


def build_model():
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


def evaluate_model(model, X_test, Y_test, category_names):
    '''
      Be aware: category_names is not used. Instead I use values. Determine if and how to use category_names.
      Problem is how to loop through them.
      '''
    Y_prediction = model.predict(X_test)

    for index, value in enumerate(Y_test):
        print('Report of the f1 score, precision and recall for category: {} \n'.format(value))
        print(classification_report(Y_test.iloc[:, index], Y_prediction[:, index]))
        print()
    pass


def save_model(model, model_filepath):
    sys.path.append(model_filepath)
    filename = 'classifier.pkl'
    pickle.dump(model, open(filename,'wb'))

    pass


def main():

    database_filepath, model_filepath = ['mysql+pymysql://{}:{}@Localhost/disasterresponse'.format(config.db_username,config.db_password),"../models"]
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print('Building model...')
    model = build_model()

    print('Training model...')
    model.fit(X_train, Y_train)

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')


if __name__ == '__main__':
    main()