# Sentiment Analysis of product reviews using Machine Learning.

## The system employs classical Natural Language Processing (NLP) techniques combined with supervised machine learning classifiers to predict whether a given product review expresses a Positive or Negative sentiment. 

## The dataset used is the publicly available Amazon Alexa Reviews dataset, which contains thousands of customer-submitted reviews along with their binary feedback labels.

Three machine learning algorithms — Naive Bayes, Logistic Regression, and Support Vector
Machine (SVM) : were trained and evaluated on the preprocessed dataset.

The bestperforming model was selected based on F1 Score and serialised using Python's pickle library.

The trained model and TF-IDF vectorizer were integrated into a Flask-based REST API, which
was subsequently deployed to the cloud platform Render.

A polished, dark-themed web interface allows end users to enter any review text and receive an instant sentiment prediction.

## Live demo: https://sentiment-analysis-po9i.onrender.com/

## Technology Stack


1. Python 3
2. NLTK
3. scikit-learn
4. Flask
5. HTML/CSS/JavaScript
6. Render (cloud deployment)
7. Google Colab (model training)

# How to Run This Project Locally

You don’t need to install anything special to run this project.

1. Download or clone this repository
2. Open the project folder
3. Double-click index.html


The app will open in your browser.

# Screenshots 

