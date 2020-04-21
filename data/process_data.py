import os
file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_dir)

import pandas as pd
import sqlalchemy as sql

# Module imports

messages = pd.read_csv('messages.csv')

categories = pd.read_csv('categories.csv')

# Drop duplicates (68 duplicate IDs removed)
categories = categories.drop_duplicates(subset=['id'])
messages = messages.drop_duplicates(subset=['id'])

# Split out categories
categories.loc[:, 'categories'] = categories['categories'].str.split(';')
categories = categories.explode('categories').rename(columns={'categories': 'category'})
categories.loc[:, 'value'] = categories['category'].str.slice(-1).astype(int)
categories.loc[:, 'category'] = categories['category'].str.slice(0, -2)

# Remove any duplicate categories, keeps 1s before 0s
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
engine = sql.create_engine('sqlite:///udacity.db')

messages.to_sql(
    name='messages',
    con=engine,
    if_exists='replace',
    index=False,
    dtype=msg_dtypes
)

categories.to_sql(
    name='categories',
    con=engine,
    if_exists='replace',
    index=False,
    dtype=cat_dtypes
)
