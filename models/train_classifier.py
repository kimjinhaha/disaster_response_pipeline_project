import sys
from sqlalchemy import create_engine

import pandas as pd

import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from sklearn.model_selection import GridSearchCV

import pickle


def load_data(database_filepath):
    """
    Load dataset from database.
    Define and return feature and target variable X and Y.
    Returns category_names.
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('df', engine)

    X = df.message
    # drop columns from df that are not dependent variables
    Y = df.drop(columns=['message', 'original', 'id', 'genre'])
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """
    A tokenization function to process the text data

    Args:
    text: list of text messages
    Returns:
    clean_tokens: tokenized text
    """

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class TextLengthExtractor(BaseEstimator, TransformerMixin):
    """
    Customized class to add the length of text as a feature.
    This class is used in building model
    """

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_length = pd.Series(X).apply(lambda x: len(x))
        return pd.DataFrame(X_length)


def build_model():
    """
    Returns a machine learning pipeline that process text and then performs
    multi-output classification on the 36 categories in the dataset.

    text_pipeline convert a collection of raw documents to a matrix of TF-IDF features.
    text_length adds the length of text as an additional feature.

    RandomForestClassifier is used to predict target variables.
    GridSearchCv is used to find the best parameters for the model.

    """

    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('text_length', TextLengthExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'features__text_pipeline__vect__stop_words': (None, 'english'),
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'features__transformer_weights': (
            {'text_pipeline': 1, 'text_length': 0.5},
            {'text_pipeline': 0.5, 'text_length': 1},
            {'text_pipeline': 0.8, 'text_length': 1},
        )
    }

    # GridSearch can take a long time
    model = GridSearchCV(pipeline, param_grid=parameters)
    # if you want to run model without GridSearch, uncomment the below and comment the above
    # model = pipeline

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model by applying trained model on the test dataset.
    Args:
    model: trained model on train dataset
    X_test, Y_test: test features and labels
    category_names: label names, which is multi-output

    Returns:
    classification_report: Per each category returns recall, precision, f-score
    """

    Y_pred = model.predict(X_test)
    Y_pred_pd = pd.DataFrame(Y_pred, columns=Y_test.columns)

    for column in Y_test.columns:
        print('------------------------------------------------------\n')
        print('FEATURE: {}\n'.format(column))
        print(classification_report(Y_test[column], Y_pred_pd[column]))


def save_model(model, model_filepath):
    """
    Save trained model as Pickle file

    Args:
    model: trained model
    model_filepath: location where the file should be saved
    """

    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
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

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()