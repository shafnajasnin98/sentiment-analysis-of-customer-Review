# Install if needed
# !pip install pandas scikit-learn nltk

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords once
nltk.download('stopwords')

# Sample dataset: reviews + label (1 = positive, 0 = negative)
data = {
    'review': [
        "This product is amazing! I love it.",
        "Terrible quality, broke after one use.",
        "Excellent value for the price.",
        "Not what I expected, very disappointing.",
        "Works perfectly, highly recommend!",
        "Worst purchase I've ever made.",
        "Good quality and fast shipping.",
        "Do not buy this product, waste of money."
    ],
    'sentiment': [1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Text preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    filtered_words = [w for w in words if w not in stop_words]
    return " ".join(filtered_words)

df['cleaned_review'] = df['review'].apply(preprocess_text)

# Features and labels
X = df['cleaned_review']
y = df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
