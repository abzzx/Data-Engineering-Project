import sys
import nltk
nltk.download(['punkt', 'wordnet'])

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    """
    Load data from database.
    Input: filepath to database 
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df.message.values
    Y = df.iloc[:, 4:]
    category_names = list(Y.columns)
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize & process text data.
    Input: text data to be cleaned and tokenized
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens

def build_model():
    """Returns model with machine learning pipeline."""
    pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
    ('clf', MultiOutputClassifier(MultinomialNB(), n_jobs=-1))
    ])

    parameters = {
        'tfidf__smooth_idf': [True, False],
        'clf__estimator__alpha': [0.001, 0.0001]}

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Returns classification report for each category classified by the model.
    Inputs:
    - model: trained model
    - X_test: test dataset features
    - Y_test: test dataset targets
    - category_names: names of all target categories to be classified
    """
    preds = model.predict(X_test)
    for idx, col in enumerate(category_names):
        print(col)
        print(classification_report(Y_test[col], preds[:, idx]), '\n')

def save_model(model, model_filepath):
    """
    Save model as a pickle file.
    Inputs:
    - model: final trained model
    - model_filepath: filepath destination to where model will be exported as pickle file
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()