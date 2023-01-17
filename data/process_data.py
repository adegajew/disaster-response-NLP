import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads messages and categories datastes, merges, and returns as dataframe

    INPUT: disaster_messages.csv and disaster_categories.csv
    OUTPUT: merged datasets as dataframe
    """

    messages = pd.read_csv(messages_filepath, header=0)
    categories = pd.read_csv(categories_filepath, header=0)

    df = messages.merge(categories, how = 'outer', on = ['id'])

    return df


def clean_data(df):
    """
    Cleans data to be ready for modelling.
    Categories are split into separate category columns.
    Category values are converted to just numbers: 0 or 1.
    Duplicates are removed.

    INPUT: dataframe with loaded data 

    OUTPUT: dataframe with clean category data
    """

    categories = df['categories'].str.split(';', expand = True)

    # first row of the categories dataframe is extracted to create names for categories
    row = categories.loc[1]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # convert category values to just numbers
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    df = df.drop('categories', axis = 1)
    df = pd.concat([df, categories], axis = 1)

    # column 'related' has some values '2'
    # those rows are excluded
    df = df[df['related'] != 2]

    df = df.drop_duplicates()

    # sanity check
    assert len(df[df.duplicated()]) == 0

    return df


def save_data(df, database_filename):
    """
    Saves the clean dataframe intoa sqlite database.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index = False, if_exists = 'replace')

    pass  


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