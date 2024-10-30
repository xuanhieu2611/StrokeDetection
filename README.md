# Stroke Prediction Model

This project applies machine learning techniques to predict the likelihood of stroke based on a person's medical history and lifestyle factors, including age, gender, BMI, heart disease, and smoking status.

## Project Overview

Using a dataset with multiple health indicators, this project trained, evaluated, and compared the performance of several machine learning models:

- **K-Nearest Neighbors (KNN)**
- **Random Forest**
- **Logistic Regression**
- **Naive Bayes**

### Dataset

The dataset used for this project includes key health indicators that have been correlated with stroke risk. Each entry in the dataset represents an individual, with features such as:

- **Age**
- **Gender**
- **BMI**
- **Medical History** (heart disease, smoking status, etc.)

## Models and Evaluation

The following models were trained and evaluated on this dataset to determine their predictive power for stroke detection.

| Model                   | Accuracy | Precision | Recall | F1-Score |
| ----------------------- | -------- | --------- | ------ | -------- |
| **K-Nearest Neighbors** | 92%      | 0.93      | 0.92   | 0.92     |
| **Random Forest**       | 97%      | 0.97      | 0.97   | 0.97     |
| **Logistic Regression** | 96%      | 0.96      | 0.96   | 0.96     |
| **Naive Bayes**         | 87%      | 0.87      | 0.87   | 0.87     |

### Confusion Matrix Results

Confusion matrices were generated to visualize the performance of each model, helping to understand the distribution of true positives, true negatives, false positives, and false negatives. Refer to the `figures` directory for detailed images of these matrices.

## Key Findings

- **Random Forest** and **Logistic Regression** models achieved the highest accuracy at 97% and 96%, respectively, showing strong predictive performance across all metrics.
- **K-Nearest Neighbors** also performed well with a 92% accuracy, though it was slightly less effective than Random Forest and Logistic Regression.
- **Naive Bayes** performed adequately, but with lower accuracy, suggesting it may be less suitable for this specific dataset.

## Installation

To run this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/xuanhieu2611/StrokeDetection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd StrokeDetection
   ```
3. Navigate to the code directory and run the main.py:
   ```bash
   cd code
   python3 main.py
   ```

## Usage

- Preprocess the data: Run the data processing script to prepare the dataset for modeling.
- Train the models: Use the model training scripts to train and evaluate each model.
- Generate Confusion Matrices: Run the evaluation script to produce confusion matrices for each model, available in the figures directory.

## Future Work

Potential next steps for the project include:

- Experimenting with other machine learning algorithms or deep learning models.
- Enhancing feature engineering to improve prediction accuracy.
- Extending the model to provide interpretability or real-time predictions.
