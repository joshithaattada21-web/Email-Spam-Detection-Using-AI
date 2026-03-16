import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# Load dataset
data = pd.read_csv("spam.csv", sep="\t", names=["label", "message"])
# Convert labels
data['label'] = data['label'].map({'ham':0, 'spam':1})
X = data['message']
y = data['label']
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Vectorize text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Prediction function
def detect_spam(email):
    email_vec = vectorizer.transform([email])
    result = model.predict(email_vec)

    if result[0] == 1:
        return "<h2 style='color:red; text-align:center;'>🚫 Spam Email</h2>"
    else:
        return "<h2 style='color:green; text-align:center;'>✅ Not Spam Email</h2>"

# Gradio UI
interface = gr.Interface(
    fn=detect_spam,
    inputs=gr.Textbox(lines=3, placeholder="Enter your email message here..."),
    outputs=gr.HTML(),
    title="<h1 style='text-align:center; font-size:45px;'>📧 Email Spam Detector</h1>",
    description="<p style='text-align:center; font-size:20px;'>Enter an email message to check whether it is Spam or Not Spam</p>",
    theme="soft",
    css="""
html, body {
    background-color: #dbeafe !important;
}

.gradio-container {
    background-color: #e0e7ff;
    border: 4px solid #4f46e5;
    border-radius: 15px;
    padding: 30px;
    margin: 30px;
}
"""
)

interface.launch()