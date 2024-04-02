import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve, GridSearchCV
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X_train, y_train = train_df['Text'], train_df['Category']
    X_test, y_test = test_df['Text'], test_df['Category']
    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, X_test):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)
    return X_train_vect, X_test_vect, vectorizer

def train_model(X_train_vect, y_train):
    classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    classifier.fit(X_train_vect, y_train)
    return classifier

def evaluate_model(classifier, X_test_vect, y_test):
    y_pred = classifier.predict(X_test_vect)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def plot_learning_curve(X_train_vect, y_train):
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000),
        X=X_train_vect,
        y=y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    
    plt.figure()
    plt.title("Learning Curves")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.plot(train_sizes, train_scores_mean, label="Training score")
    plt.plot(train_sizes, validation_scores_mean, label="Cross-validation score")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('./results/learning_curves.png')
    print("Learning curve saved successfully.")

def save_models(classifier, vectorizer):
    joblib.dump(classifier, './models/news_classifier.pkl')
    joblib.dump(vectorizer, './models/tfidf_vectorizer.pkl')
    print("Models saved successfully.")

def main():
    train_path = './data/bbc_news_train.csv'
    test_path = './data/bbc_news_tests.csv'
    X_train, y_train, X_test, y_test = load_data(train_path, test_path)
    X_train_vect, X_test_vect, vectorizer = preprocess_data(X_train, X_test)
    classifier = train_model(X_train_vect, y_train)
    evaluate_model(classifier, X_test_vect, y_test)
    plot_learning_curve(X_train_vect, y_train)
    save_models(classifier, vectorizer)

if __name__ == "__main__":
    main()
