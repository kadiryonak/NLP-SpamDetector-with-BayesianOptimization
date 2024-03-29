# Spam Detection Project

## Overview
This project aims to detect spam messages using natural language processing (NLP) techniques and deep learning. The core of the project is built with Python, utilizing libraries such as Pandas, NumPy, NLTK (Natural Language Toolkit), TensorFlow, and Scikit-learn. The process includes data cleaning, preprocessing, feature extraction using TF-IDF vectorization, and classification using a LSTM neural network model. The project also incorporates Bayesian Optimization to fine-tune the LSTM model parameters for improved accuracy.

## Project Structure
To maintain a clean and manageable codebase, the project is structured into several key sections, organized into separate scripts/files for better modularity and readability:

### 1. Data Preprocessing (`data_preprocessing.py`)
This script includes:
- Loading and encoding the dataset
- Data cleaning (removing duplicates and handling missing values)
- Text preprocessing (lowercasing, removing punctuation and numbers, tokenization, removing stopwords, lemmatization, and stemming)

### 2. Feature Extraction (`feature_extraction.py`)
This script focuses on converting text data into a numerical format using TF-IDF vectorization, making it suitable for machine learning models.

### 3. Model Training and Optimization (`model_training_optimization.py`)
Contains the implementation of the LSTM model, including:
- Model definition and compilation
- Model training with early stopping to prevent overfitting
- Bayesian Optimization to find the best hyperparameters for the LSTM model

### 4. Model Evaluation (`model_evaluation.py`)
Evaluates the trained model on a test set to assess its performance, specifically its accuracy in classifying messages as spam or not spam.

### 5. Utilities (`utils.py`)
Includes utility functions that might be used across different parts of the project, such as loading data, tokenization, and padding sequences.

## Dependencies
The project requires the following Python libraries:
- pandas
- numpy
- re
- nltk
- sklearn
- tensorflow
- bayes_opt

Ensure that you have these installed in your environment before running the project.

## Installation
Clone this repository to your local machine using:
```bash
git clone <repository-url>

