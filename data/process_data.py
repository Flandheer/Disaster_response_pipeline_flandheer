import os


def load_data(messages_filepath, categories_filepath):
    '''
    Function to load and structure disaster categories and message in single DataFrame.
    INPUT
    :param messages_filepath: Filepath to get csv file messages
    :param categories_filepath: Filepath to get csv file categories
    OUTPUT
    :return: single dataframe with messages and categories
    '''

    #import libraries
    import pandas as pd

    #import dataset
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
    from sqlalchemy import create_engine
    import config

    os.listdir()
    db_connection = 'mysql+pymysql://{}:{}@Localhost/disasterresponse'.format(config.db_username,config.db_password)
    db_disaster_response = create_engine(db_connection)
    df.to_sql(database_filename, con=db_disaster_response, if_exists='replace', index=False)

    return None

def main():

    messages_filepath, categories_filepath, database_filepath = ['disaster_messages.csv', 'disaster_categories.csv', 'disasterresponse']

    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
          .format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)

    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)

    print('Cleaned data saved to database!')
    


if __name__ == '__main__':
    main()
