'''Ingests the data generated by process_data.py, uses it to train and evaluate a
classifier object. Saves the object and generated metrics to disk.
'''

import os
import pickle
import string
import sys
from functools import lru_cache

import nltk
import pandas as pd
import sqlalchemy as sql
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# pylint: disable=redefined-outer-name

file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_dir)

def load_data(db_path):
    '''Load processed data from the specified SQLite database, return
    the datasets required for modeling.

    Returns
    X - np.array
    y - np.array
    cat_names - list
    '''

    db_path = os.path.realpath(db_path)
    db_conn = sql.create_engine(f'sqlite:///{db_path}')

    messages = pd.read_sql_table('messages', db_conn, index_col='id')
    categories = pd.read_sql_table('categories', db_conn, index_col='id')

    matches = sum(messages.index == categories.index)
    msg_count = len(messages.index)
    cat_count = len(categories.index)

    # Check that record IDs are aligned between datasets
    assert matches == msg_count == cat_count, "Error: datasets are not aligned."

    X = messages['message'].to_numpy()
    y = categories.to_numpy()
    cat_names = categories.columns.tolist()

    return X, y, cat_names


class Tokenizer(BaseEstimator, TransformerMixin):
    '''Custom tokenizer implementation. For a given string input,
    the following transformations are applied:

    Set to lowercase
    Tokenize by word
    Lemmatize each word
    Remove stopwords
    '''

    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.stop = stopwords.words('english')

    def _get_wordnet_pos(self, treebank_tag):
        '''Turns a treebank tag (generated by nltk) into a wordnet tag which
        can be used by the lemmatzier'''

        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # Default parameter
            return wordnet.NOUN

    @lru_cache(maxsize=16192)
    def _tokenize(self, comment):
        comment = comment.lower()
        tokens = nltk.word_tokenize(comment)
        tokens = [x for x in tokens
                  if x
                  and x not in string.punctuation]

        try:
            pos_tags = nltk.pos_tag(tokens)
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
            pos_tags = nltk.pos_tag(tokens)

        assert len(tokens) == len(pos_tags), 'Error: Mismatched tags and tokens'

        # Remove stopwords after POS tagging to avoid changing meaning
        tokens = [self.wnl.lemmatize(token,
                                     pos=self._get_wordnet_pos(pos_tag))
                  for token, pos_tag
                  in pos_tags
                  if token not in self.stop]

        return tokens

    def __call__(self, comment):
        return self._tokenize(comment)



def build_model():
    '''Create a pipeline which will automatically select hyperparameters for
    two candidate algorithms upon fitting to a training dataset.
    '''

    pipe = Pipeline([
        ('features', CountVectorizer(tokenizer=Tokenizer(),
                                     lowercase=False,
                                     strip_accents='unicode')),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])

    param_grids = [
        {'clf__estimator': [RandomForestClassifier(random_state=42, n_jobs=4)],
         'clf__estimator__n_estimators': [16, 32, 64, 128],
         'clf__estimator__max_features': ['sqrt', 'log2', None],
         'clf__estimator__max_samples': [0.25, 0.5, 0.75, None]},
        {'clf__estimator': [XGBClassifier(random_state=42,
                                          n_jobs=4)],
         'clf__estimator__learning_rate': [0.1, 0.01, 0.001],
         'clf__estimator__booster': ['gbtree', 'gblinear', 'dart'],
         'clf__estimator__colsample_bytree': [0.25, 0.5, 0.75, 1.0],
         'clf__estimator__num_parallel_tree': [1, 4, 16]}
    ]

    optimizer = RandomizedSearchCV(
        pipe,
        param_grids,
        cv=3,
        verbose=2
    )

    return optimizer


def evaluate_model(model, X_val, y_val, cat_names):
    '''Use the provided model to make predictions on the validation dataset,
    returns a dataframe containing f1, precision and recall scores for each
    predicted category
    '''

    y_pred = model.predict(X_val)

    records = []
    for inx, label in enumerate(cat_names):
        true = y_val[:, inx]
        pred = y_pred[:, inx]
        record = dict(
            category=label,
            f1=f1_score(true, pred),
            precision=precision_score(true, pred),
            recall=recall_score(true, pred)
        )
        records.append(record)

    performance = pd.DataFrame.from_records(records)
    performance.to_excel('model_performance.xlsx', index=False)

    return performance


def save_model(model, params, model_path):
    '''Save the provided model object to model_filepath'''

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    param_path = f"{model_path.split('.')[0]}_params.pkl"
    with open(param_path, 'wb') as f:
        pickle.dump(params, f)


def main():
    '''As defined by template'''

    if len(sys.argv) == 3:
        db_path, model_path = sys.argv[1:]
        # print('Loading data...\n    DATABASE: {}'.format(db_path))
        X, y, cat_names = load_data(db_path)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        optimizer = build_model()

        print('Training model...')
        optimizer.fit(X_train, y_train)

        model = optimizer.best_estimator_
        params = optimizer.best_params_

        print('Evaluating model...')
        _ = evaluate_model(model, X_val, y_val, cat_names)

        print('Saving model...\n    MODEL: {}'.format(model_path))
        save_model(model, params, model_path)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()