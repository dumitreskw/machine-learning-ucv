import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Preprocess data (e.g., using NLTK or spaCy)
# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data['review'], train_data['label'], test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)

# Initialize and train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the validation set
y_pred = nb_classifier.predict(X_val_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
report = classification_report(y_val, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Predict labels for the test set
X_test_tfidf = tfidf_vectorizer.transform(test_data['review'])
test_data['predicted_label'] = nb_classifier.predict(X_test_tfidf)

# Save the solution CSV
test_data.to_csv('solution.csv', columns=['review', 'predicted_label'], index=False)