import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("spam.csv", sep="\t", names=["label", "message"])

# Convert labels into numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split data into input and output
X = data['message']
y = data['label']
# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert text into numbers
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict test data
predictions = model.predict(X_test_vec)

# Print accuracy
print("Model Accuracy:", accuracy_score(y_test, predictions))

# Test with user input
email = input("Enter email message: ")

email_vec = vectorizer.transform([email])
result = model.predict(email_vec)

if result[0] == 1:
    print("Spam Email 🚫")
else:
    print("Not Spam Email ✅")