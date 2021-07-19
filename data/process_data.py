import sys
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Function to load and structure disaster categories and message in single DataFrame.
    INPUT
    :param messages_filepath: Filepath to get csv file messages
    :param categories_filepath: Filepath to get csv file categories
    OUTPUT
    :return: single dataframe with messages and categories
    '''

    # import libraries
    import pandas as pd

    # import dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge datasets into one dataframe
    df = messages.merge(categories, left_on='id', right_on='id')

    # Create column names for categories, which aren't neatly structured in dataset
    categories = categories['categories'].str.split(';', expand=True)
    category_colnames = list(categories.iloc[0].str.slice(0, -2))
    categories.columns = category_colnames

    # Create values for categories, which aren't neatly structured in dataset
    for column in categories:
        categories[column] = categories[column].str.split('-').str[1]
        categories[column] = categories[column].astype(str)

    # Drop colomn categories and add ordered columns to DataFrame
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)

    return df


def clean_data(df):
    '''
    Clean data by removing duplicates
    INPUT
    :param df: dataframe from load_data()
    OUTPUT
    :return: cleaned data
    '''

    df = df.drop_duplicates()
    df = df.dropna(subset=['request'])

    return df


def save_data(df, database_filename):
    '''

    :param df: Dataframe to store in a database
    :param database_filename: reference to use for database name
    :return: sqlite database with cleaned data
    '''

    # Create engine for a connection with sqlite database
    db_disaster_response = create_engine('sqlite:///{}'.format(database_filename))

    # Store dataframe as sqlite database in
    df.to_sql('Messages', db_disaster_response, index=False, if_exists='replace')

    return None


def main():
    '''
    Main code to run ETL module. To run the following arguments should be given
        disaster_messages.csv disaster_categories.csv DisasterResponse.db
    Add tot parameters in your IDE or in your command line following
        python3 process_data.py
    :return: Cleaned data stored in a sqlite database
    '''

    if len(sys.argv) == 4:  # Arguments are stored in parameters of text editor

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()