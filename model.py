import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Load Sentiment140 data
df = pd.read_csv('training.1600000.processed.noemoticon.csv.zip', encoding='latin-1', header=None)
df = df[[0, 5]]
df.columns = ['polarity', 'text']
print(df.head())

# Keep only positive (4) and negative (0)
df = df[df.polarity != 2]
df['polarity'] = df['polarity'].map({0: 0, 4: 1})
print(df['polarity'].value_counts())

# Simple cleaning function
def clean_text(text):
    return text.lower()

df['clean_text'] = df['text'].apply(clean_text)

print(df[['text', 'clean_text']].head())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['polarity'], test_size=0.2, random_state=42)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

# Vectorize text using TF-IDF (using binary features for BernoulliNB)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("TF-IDF shape (train):", X_train_tfidf.shape)
print("TF-IDF shape (test):", X_test_tfidf.shape)


# Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train_tfidf, y_train)
bnb_pred = bnb.predict(X_test_tfidf)
print("Bernoulli Naive Bayes Accuracy:", accuracy_score(y_test, bnb_pred))

# Support Vector Machine (LinearSVC)
svm = LinearSVC(max_iter=1000)
svm.fit(X_train_tfidf, y_train)
svm_pred = svm.predict(X_test_tfidf)
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))

# Logistic Regression
logreg = LogisticRegression(max_iter=100)
logreg.fit(X_train_tfidf, y_train)
logreg_pred = logreg.predict(X_test_tfidf)
print("Logistic Regression Accuracy:", accuracy_score(y_test, logreg_pred))

# Optional: Detailed classification report for each model
print("\nBernoulliNB Classification Report:\n", classification_report(y_test, bnb_pred))
print("\nSVM Classification Report:\n", classification_report(y_test, svm_pred))
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, logreg_pred))

# Test on custom tweets
sample_tweets = ["I love this!", "I hate that!", "It was okay, not great."]
sample_vec = vectorizer.transform(sample_tweets)

print("\nSample Predictions:")
print("BernoulliNB:", bnb.predict(sample_vec))
print("SVM:", svm.predict(sample_vec))
print("Logistic Regression:", logreg.predict(sample_vec))
