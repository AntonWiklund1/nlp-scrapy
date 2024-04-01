import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the training and test data
train_df = pd.read_csv('./data/bbc_news_train.csv')
test_df = pd.read_csv('./data/bbc_news_tests.csv')

# Assuming the structure is ['text', 'label'] where 'text' is the article and 'label' is the category
X_train = train_df['Text']
y_train = train_df['Category']
X_test = test_df['Text']
y_test = test_df['Category']


vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)


classifier = LinearSVC()
classifier.fit(X_train_vect, y_train)


y_pred = classifier.predict(X_test_vect)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model as a .pkl file
try:
    joblib.dump(classifier, './models/news_classifier.pkl')
    joblib.dump(vectorizer, './models/tfidf_vectorizer.pkl')
    print("Models saved successfully.")
except Exception as e:
    print(f"Failed to save the model: {str(e)}")