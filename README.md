

# Spam Mail Detection System
Spam detection using ML and Flask


A Machine Learning and Natural Language Processing (NLP) based application that classifies SMS and email messages as Spam or Ham (Legitimate). The system uses text preprocessing, TF-IDF feature extraction, and machine learning classification algorithms to detect unwanted messages. The selected model is deployed using Flask to provide real-time spam prediction through a web interface.

## Project Overview

Spam messages are a major challenge in digital communication due to phishing attempts, fraud, and unwanted advertisements. This project develops an automated spam detection system that learns patterns from labeled message data and predicts whether a new message is spam or legitimate.

The project follows an end-to-end machine learning workflow including data preprocessing, feature extraction, model training, evaluation, model comparison, and deployment.

## Machine Learning Pipeline

```
Raw Text Data
      ↓
Text Preprocessing
      ↓
TF-IDF Feature Extraction
      ↓
Model Training
      ↓
Model Comparison & Evaluation
      ↓
Best Model Selection
      ↓
Flask Deployment
      ↓
Spam/Ham Prediction
```

## NLP Processing

The text data is processed before training to improve model performance.

Preprocessing steps include:

- Lowercase conversion
- Removal of unnecessary characters
- Text normalization

The processed text is converted into numerical feature vectors using TF-IDF (Term Frequency-Inverse Document Frequency), which identifies the importance of words within the dataset.

## Machine Learning Models

Two supervised machine learning algorithms were implemented and evaluated for spam message classification.

### Multinomial Naive Bayes

Multinomial Naive Bayes is a probability-based classification algorithm widely used for text classification tasks.

It works effectively with TF-IDF features by analyzing word occurrence patterns and calculating the probability of a message belonging to the Spam or Ham category.

Advantages:

- Performs well on text-based datasets
- Fast training and prediction
- Efficient with high-dimensional feature spaces

### Logistic Regression

Logistic Regression is a supervised learning algorithm used for binary classification problems.

In this project, it uses TF-IDF feature vectors to estimate the probability of a message being Spam or Ham.

Advantages:

- Simple and interpretable
- Effective for text classification
- Provides strong baseline performance

## Model Comparison and Selection

Both Multinomial Naive Bayes and Logistic Regression models were trained and evaluated using the same dataset and TF-IDF feature extraction technique.

The models were compared using:

- Accuracy
- Precision
- Recall
- F1 Score

Based on the evaluation results, the model with the highest performance was selected as the final classifier and integrated into the Flask application for real-time prediction.

## Application Architecture

The application consists of the following components:

User Interface:
Flask-based web interface where users can enter messages.

Preprocessing Layer:
Cleans and prepares input text before prediction.

Feature Extraction Layer:
Transforms processed text into TF-IDF numerical vectors.

Prediction Layer:
Uses the selected machine learning model for classification.

Output Layer:
Displays the final Spam or Ham prediction result.

## Technologies Used

Programming Language:
Python

Machine Learning:
Scikit-learn, Multinomial Naive Bayes, Logistic Regression

Natural Language Processing:
TF-IDF Vectorization, NLTK

Web Framework:
Flask

Frontend:
HTML, CSS

Libraries:
Pandas, NumPy, Joblib

## Project Structure

```
spam-mail-detection/
│
├── app.py # Flask application for real-time spam prediction
├── train_model.py # Script for training and saving the ML model
├── preprocess.py # Text preprocessing and cleaning functions
├── requirements.txt # Required Python dependencies
│
├── model/
│ ├── spam.pkl # Trained spam classification model
│ └── vectorizer.pkl # Saved TF-IDF vectorizer
│
├── dataset/
│ └── SMSSpamCollection # Dataset containing spam and ham messages
│
├── templates/
│ └── index.html # Web interface for spam prediction
```

## Sample Prediction

Input:

```
Congratulations! You have won a free prize. Claim now.
```

Output:

```
Spam
```

Input:

```
Can we meet tomorrow for the project discussion?
```

Output:

```
Ham
```

## Conclusion

This project demonstrates the practical application of Natural Language Processing and Machine Learning for automated spam classification.

By combining TF-IDF feature extraction, model comparison, machine learning classification, and Flask deployment, the system provides an efficient solution for real-time spam detection.

## Author

**Meesala Nikhita**

GitHub:  
https://github.com/Meesala-Nikhita

LinkedIn:  
https://www.linkedin.com/in/nikhita-meesala-4113a131/
