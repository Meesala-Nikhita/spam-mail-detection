import re
import string

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove phone numbers (10+ digits)
    text = re.sub(r'\b\d{10,}\b', '', text)

    # Remove remaining numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Remove extra spaces, tabs, and new lines
    text = re.sub(r'\s+', ' ', text).strip()

    return text
