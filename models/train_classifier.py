import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import nltk
nltk.download('punkt', 'stopwords', 'wordnet', 'omw-1.4')

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import pickle


def load_data(database_filepath):
    """
    Loads data from the database and creates two data frames: X, Y.
    INPUT: filepath to SQLite database.
    OUTPUT: X - data frame storing input variable: message
            Y - data frame storing output variables: 36 categories
            category_names - names of 36 categories
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('messages', con = engine)

    X = df['message'] 
    Y = df.iloc[:, 4:]
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """
    Processes text data.
    Includes: punctuation removal, tokenization, stop word removal,
            and lemmatization.

    INPUT: text data to be processed
    OUTPUT: text data in a format ready for modelling
    """
    # punctuation removal:
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    # tokenization:
    tokens = word_tokenize(text)
    # stop word removal:
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    # lemmatization:
    #stem = PorterStemmer() # both stemming and lemmatization should not be used
    lem = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens: 
        clean_tok = lem.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)   
    
    return clean_tokens


def build_model():
    """
    Pipeline with ML model to classify input data.
    Algorithm used is AdaBoostClassifier with DecisionTreeClassifier as base estimator.
    Grid Search CV on selected parameters.
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(base_estimator = DecisionTreeClassifier()))),
    ])

    parameters = {
        'clf__estimator__base_estimator__max_depth': [1, 2]
        #'clf__estimator__base_estimator__min_samples_split': [2, 5],
        #'clf__estimator__learning_rate': [0.01, 0.1, 1.0, 1.2]
    }

    model = GridSearchCV(pipeline, param_grid = parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Prints classification reports in summary and detailed form.
    INPUT: model, X test data, Y test data, category names taken from Y data
    """

    y_pred = model.predict(X_test)

    # summary classification report
    print(classification_report(Y_test, y_pred, target_names = category_names, zero_division = 0))

    # detailed classification report
    for i in range(len(category_names)):
        print(category_names[i])
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i], zero_division = 0))

    pass


def save_model(model, model_filepath):
    """
    Exports the model as a pickle file.
    INPUT: model and a filepath where the model should be saved.
    """

    with open(model_filepath,'wb') as f:
        pickle.dump(model,f)

    pass


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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()