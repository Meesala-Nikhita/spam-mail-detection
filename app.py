from flask import Flask, render_template, request # type: ignore
import pickle
from preprocess import preprocess_text

app = Flask(__name__)

# Load model and vectorizer
with open("model/spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    # Get message from form
    message = request.form.get('message')

    # Safety check
    if not message:
        return render_template(
            "index.html",
            label="No Message Entered",
            confidence="0"
        )

    # Clean text
    cleaned_message = preprocess_text(message)

    # Convert into vector
    vector = vectorizer.transform([cleaned_message])

    # Predict
    result = model.predict(vector)
    probabilities = model.predict_proba(vector)[0]

    spam_probability = probabilities[1] * 100
    ham_probability = probabilities[0] * 100

    # Decide label
    if result[0] == 1:
        label = "SPAM"
        confidence = spam_probability
    else:
        label = "NOT SPAM"
        confidence = ham_probability

    return render_template(
        "index.html",
        label=label,
        confidence=f"{confidence:.2f}",
        message=message
    )

if __name__ == "__main__":
    # app.run(debug=True)
    app.run(debug=False)