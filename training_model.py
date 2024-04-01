import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the training and test data
train_df = pd.read_csv('./data/bbc_news_train.csv')
test_df = pd.read_csv('./data/bbc_news_tests.csv')


X_train = train_df['Text']
y_train = train_df['Category']
X_test = test_df['Text']
y_test = test_df['Category']

vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

classifier = LinearSVC()
classifier.fit(X_train_vect, y_train)

# Generate learning curve data
train_sizes, train_scores, validation_scores = learning_curve(
    estimator=LinearSVC(),
    X=X_train_vect,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Calculate the mean and standard deviation of the training and validation scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
validation_scores_mean = np.mean(validation_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)

# Plot the learning curve
plt.figure()
plt.title("Learning Curves")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std,
                 validation_scores_mean + validation_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, validation_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.legend(loc="best")
plt.grid()

try:
    plt.savefig('./results/learning_curves.png')
    print("Learning curve saved successfully.")
except Exception as e:
    print(f"Failed to save the learning curve: {str(e)}")

# Evaluate the model on the test set
y_pred = classifier.predict(X_test_vect)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model and vectorizer as .pkl files
try:
    joblib.dump(classifier, './models/news_classifier.pkl')
    joblib.dump(vectorizer, './models/tfidf_vectorizer.pkl')
    print("Models saved successfully.")
except Exception as e:
    print(f"Failed to save the model: {str(e)}")
