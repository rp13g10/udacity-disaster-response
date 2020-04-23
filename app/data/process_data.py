'''Ingests data from the provided csv files, parses them into a format which can be used
for model creation and saves the output to a SQLite database.'''

import os
import sys

import pandas as pd
import sqlalchemy as sql

# File imports
if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(file_dir)

def load_data(msg_path, cat_path):
    '''Load data from the specified file paths'''

    messages = pd.read_csv(msg_path)
    categories = pd.read_csv(cat_path)
    return messages, categories


def clean_data(messages, categories):
    '''Parse provided datasets into a suitable format for use by scikit-learn'''

    # Drop duplicates (68 duplicate IDs removed)
    categories = categories.drop_duplicates(subset=['id'])
    messages = messages.drop_duplicates(subset=['id'])

    # Split out categories
    categories.loc[:, 'categories'] = categories['categories'].str.split(';')
    categories = categories.explode('categories').rename(columns={'categories': 'category'})
    categories.loc[:, 'value'] = categories['category'].str.slice(-1).astype(int)
    categories.loc[:, 'category'] = categories['category'].str.slice(0, -2)

    # Remove any duplicate categories, keeping 1s before 0s
    categories.loc[:, 'value'] = categories['value'].map(lambda x: int(x >= 1))
    categories = categories.sort_values(by=['id', 'category', 'value'])
    categories = categories.drop_duplicates(subset=['id', 'category'], keep='last')

    # Pivot from long to wide form
    categories = pd.pivot_table(
        categories,
        index=['id'],
        columns=['category'],
        aggfunc='sum',
        fill_value=0)
    categories.columns = categories.columns.droplevel(0)
    categories = categories.reset_index()

    return messages, categories


def save_data(messages, categories, db_path):
    '''Save the processed datasets down to the provided database file'''

    # Set up dtype mappings
    msg_dtypes = {
        'id': sql.types.Integer,
        'message': sql.types.Unicode
    }

    cat_dtypes = {
        'id': sql.types.Integer
    }

    cat_dtypes.update({
        col: sql.types.Integer
        for col in categories.columns
        if col != 'id'})

    # Upload data to SQLite
    db_path = os.path.realpath(db_path)
    db_conn = sql.create_engine(f'sqlite:///{db_path}')

    messages.to_sql(
        name='messages',
        con=db_conn,
        if_exists='replace',
        index=False,
        dtype=msg_dtypes
    )

    categories.to_sql(
        name='categories',
        con=db_conn,
        if_exists='replace',
        index=False,
        dtype=cat_dtypes
    )


def main():
    '''As defined by template, executes the functions defined above'''

    # pylint: disable=unbalanced-tuple-unpacking
    if len(sys.argv) == 4:

        msg_path, cat_path, db_path = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(msg_path, cat_path))
        messages, categories = load_data(msg_path, cat_path)

        print('Cleaning data...')
        messages, categories = clean_data(messages, categories)

        print('Saving data...\n    DATABASE: {}'.format(db_path))
        save_data(messages, categories, db_path)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'messages.csv categories.csv udacity.db')


if __name__ == '__main__':
    main()
