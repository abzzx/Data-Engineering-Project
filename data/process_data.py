import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Returns a dataframe of messages & categories datasets.
    
    Inputs:
    - messages_filepath: filepath to messages dataset
    - categories_filepath: filepath to categories dataset
    """ 
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories)
    return df


def clean_data(df):
    """
    Returns cleaned dataframe where:
        - Values in the categories column are split so that each value becomes a separate column.
        - Columns of categories are renamed new column names.
        - String values encoded into numeric values of 0 or 1.
        - Duplicates dropped.
        
    Input: dataframe to be cleaned
    """
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])
    
    categories.related[categories.related == 2]  = 1

    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1, join='inner')
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    """
    Saves the dataframe into an sqlite database.
    Inputs:
    - df: dataframe to be saved
    - database_filename: filepath to database
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False)
    

def main():
    if len(sys.argv) == 4:

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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()