import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from preprocess import preprocess_text

# Load Dataset
df = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Convert Labels
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Clean Messages
df["message"] = df["message"].apply(preprocess_text)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# MODEL 1: NAIVE BAYES

nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_pred = nb_model.predict(X_test_vec)

nb_accuracy = accuracy_score(y_test, nb_pred)
nb_precision = precision_score(y_test, nb_pred)
nb_recall = recall_score(y_test, nb_pred)
nb_f1 = f1_score(y_test, nb_pred)


print("MULTINOMIAL NAIVE BAYES")

print("Accuracy :", nb_accuracy)
print("Precision:", nb_precision)
print("Recall   :", nb_recall)
print("F1 Score :", nb_f1)

print("Confusion Matrix:")
print(confusion_matrix(y_test, nb_pred))


# MODEL 2: LOGISTIC REGRESSION

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)
lr_pred = lr_model.predict(X_test_vec)

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

print("LOGISTIC REGRESSION")

print("Accuracy :", lr_accuracy)
print("Precision:", lr_precision)
print("Recall   :", lr_recall)
print("F1 Score :", lr_f1)

print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))


# SELECT BEST MODEL
if lr_f1 > nb_f1:
    best_model = lr_model
    best_name = "Logistic Regression"
else:
    best_model = nb_model
    best_name = "Multinomial Naive Bayes"

print(f"\nBest Model Selected: {best_name}")

# Save Best Model
pickle.dump(
    best_model,
    open("model/spam_model.pkl", "wb")
)

pickle.dump(
    vectorizer,
    open("model/vectorizer.pkl", "wb")
)

print("\nModel and Vectorizer saved successfully.")