import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

class StrokeRiskPredictor:
    def __init__(self, data_path):
        # Load the dataset
        self.dataset = pd.read_csv(data_path)

    def preprocess_data(self):
        # Convert categorical variables into numerical representations using one-hot encoding.
        # This will create new binary columns for each unique value in 'gender', 'ever_married',
        # 'work_type', 'Residence_type', and 'smoking_status'. This step is necessary for many machine
        # learning models, which cannot work directly with categorical data.
        dataset_stroke_prediction_encoded = pd.get_dummies(self.dataset, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])

        # Replace any missing values in the 'bmi' column with the median value of 'bmi' from the dataset.
        # Filling missing values with the median ensures that we handle incomplete data without introducing
        # extreme outliers or biases that could arise from using other imputation strategies like mean.
        dataset_stroke_prediction_encoded['bmi'].fillna(dataset_stroke_prediction_encoded['bmi'].median(), inplace=True)

        # Select the feature variables (X) by excluding the 'stroke' column, which is the target variable
        # we are trying to predict, and the 'id' column, which is just an identifier and holds no predictive
        # power. This ensures the model only trains on relevant features.
        X = dataset_stroke_prediction_encoded.drop(columns=['stroke', 'id'])

        # Define the target variable (y), which we are trying to predict. In this case, 'stroke' indicates
        # whether a person has had a stroke (1) or not (0). This is the outcome we want to classify.
        y = dataset_stroke_prediction_encoded['stroke']

        # Initialise SMOTE (Synthetic Minority Over-sampling Technique), which is used to address the issue
        # of class imbalance in the dataset. In this case, the number of stroke cases (minority class) is much lower
        # than the number of non-stroke cases (majority class). SMOTE generates synthetic samples to balance the two
        # classes, preventing the model from being biased towards the majority class.
        smote = SMOTE(random_state=42)

        # Apply SMOTE to resample the feature set (X) and the target variable (y). This step will balance the
        # dataset by generating new instances of the minority class (stroke cases), ensuring that both classes
        # (stroke and no stroke) are equally represented in the training data. This helps improve model performance
        # when predicting the minority class.
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Return the resampled features (X_resampled) and target variable (y_resampled). This dataset will be used
        # to train the machine learning model, ensuring a balanced dataset that can better predict stroke cases.
        return X_resampled, y_resampled
    
    def create_training_and_testing_data(self, x, y):
        # Split the data into training and testing sets using train_test_split.
        # 'x' represents the features, and 'y' represents the target (labels).
        # We are splitting the data such that 70% is used for training and 30% for testing.
        # 'random_state=42' ensures that the split is reproducible every time the code is run.
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        # Return the split datasets: X_train and y_train are used to train the model,
        # while X_test and y_test are used to evaluate the model's performance on unseen data.
        return X_train, X_test, y_train, y_test

    def k_nearest_neighbours_classifier(self, X_train, X_test, y_train, y_test):
        # Initialize the k-Nearest Neighbors classifier with 2 neighbors.
        # 'n_neighbors=2' means the algorithm will look at the 2 nearest data points to classify a new data point.
        model = KNeighborsClassifier(n_neighbors=2)

        # Fit the k-NN model on the training data. This trains the model by storing the data points
        # in memory, which will be used during the prediction phase to find the closest neighbors.
        model.fit(X_train, y_train)

        # Use the trained model to predict the class labels for the test data.
        # 'X_test.values' is used to ensure the input is in the correct format (NumPy array) for prediction.
        y_pred = model.predict(X_test.values)

        # Print the classification report which provides detailed metrics like precision, recall, f1-score,
        # and accuracy for each class. These metrics help evaluate the performance of the classifier.
        # - Precision: Proportion of true positives out of all predicted positives.
        # - Recall: Proportion of true positives out of all actual positives.
        # - F1-Score: Harmonic mean of precision and recall, balancing the two.
        # - Accuracy: Overall correctness of the model.
        print("K nearest neighbours")
        print(classification_report(y_test, y_pred))

        # Create a confusion matrix, which compares the actual values with predicted values.
        # It shows the number of true positives, true negatives, false positives, and false negatives.
        # This matrix helps identify how well the model is distinguishing between the classes.
        cm = confusion_matrix(y_test, y_pred)

        # Create a confusion matrix display object and set the labels for the two classes (0 and 1).
        # The display will visually represent the confusion matrix as a plot, with a color map for easier interpretation.
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

        # Plot the confusion matrix with a blue color map ('Blues') for better visual distinction.
        # Adding a title 'Confusion Matrix' to the plot for clarity.
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix")

        # Save and display the confusion matrix plot on the screen.
        fname = Path("..", "figs", "KNN_ConfusionMatrix.pdf")
        plt.savefig(fname)
        # plt.show()

    def random_forest_classifier(self, X_train, X_test, y_train, y_test):
        # Initialize the Random Forest classifier with the following parameters:
        # - random_state=42: Sets the seed for reproducibility, ensuring consistent results across runs.
        # - class_weight='balanced': Adjusts the weights for each class based on their frequency in the data.
        #   This helps handle class imbalances by giving more importance to the minority class.
        model = RandomForestClassifier(random_state=42, class_weight='balanced')

        # Train the Random Forest classifier on the training data (X_train, y_train).
        # The model learns by creating multiple decision trees and averaging their predictions to minimize overfitting.
        model.fit(X_train, y_train)

        # Use the trained Random Forest model to make predictions on the test data (X_test).
        # The predicted values (y_pred) are the model's classification output for each sample in the test set.
        y_pred = model.predict(X_test)

        # Print a classification report that provides detailed metrics such as precision, recall, f1-score,
        # and accuracy for the model's predictions. This helps evaluate how well the model performed.
        print("Random Forest")
        print(classification_report(y_test, y_pred))

        # Generate a confusion matrix that compares the actual test labels (y_test) with the predicted labels (y_pred).
        # It helps visualize how many samples were correctly or incorrectly classified.
        cm = confusion_matrix(y_test, y_pred)

        # Display the confusion matrix in a plot, with labels for both classes (0 and 1).
        # The color map 'Blues' provides a visual representation of the matrix, making it easier to interpret.
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap='Blues')

        # Set the plot's title to "Confusion Matrix" for clarity.
        plt.title("Confusion Matrix")

        # Save and display the confusion matrix plot on the screen.
        fname = Path("..", "figs", "RandomForest_ConfusionMatrix.pdf")
        plt.savefig(fname)
        # plt.show()

    def logistic_regression_classifier(self, X_train, X_test, y_train, y_test):
        # Initialize the Logistic Regression classifier with the following parameters:
        # - max_iter=4000: Maximum number of iterations for optimization.
        # - random_state=42: Seed for reproducibility.
        # - class_weight='balanced': Adjusts class weights to handle imbalanced data.
        model = LogisticRegression(max_iter=4000, random_state=42, class_weight='balanced')

        # Train the Logistic Regression model on the training data (X_train, y_train).
        # The model learns the relationship between the input features and the target variable.
        model.fit(X_train, y_train)

        # Use the trained Logistic Regression model to predict the target values for the test data (X_test).
        # The predicted values (y_pred) represent the model's classification output for each sample in the test set.
        y_pred = model.predict(X_test)

        # Print a classification report that provides detailed metrics such as precision, recall, f1-score,
        # and accuracy for the model's predictions. This helps evaluate how well the model performed.
        print("Logistic Regression")
        print(classification_report(y_test, y_pred))

        # Generate a confusion matrix that compares the actual test labels (y_test) with the predicted labels (y_pred).
        # It helps visualize how many samples were correctly or incorrectly classified.
        cm = confusion_matrix(y_test, y_pred)

        # Display the confusion matrix in a plot, with labels for both classes (0 and 1).
        # The color map 'Blues' provides a visual representation of the matrix, making it easier to interpret.
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap='Blues')

        # Set the plot's title to "Confusion Matrix" for clarity.
        plt.title("Confusion Matrix")

        # Save and display the confusion matrix plot on the screen.
        fname = Path("..", "figs", "LogisticRegression_ConfusionMatrix.pdf")
        plt.savefig(fname)
        # plt.show()

    def bernoulli_naive_bayes(self, X_train, X_test, y_train, y_test):
        # Initialize the Bernoulli Naive Bayes classifier.
        model = BernoulliNB()

        # Train the Naive Bayes model on the training data (X_train, y_train).
        # The model learns the probability distribution of the features for each class.
        model.fit(X_train, y_train)

        # Use the trained Naive Bayes model to predict the target values for the test data (X_test).
        # The predicted values (y_pred) represent the model's classification output for each sample in the test set.
        y_pred = model.predict(X_test)

        # Print a classification report that provides detailed metrics such as precision, recall, f1-score,
        # and accuracy for the model's predictions. This helps evaluate how well the model performed.
        print("Naives Bayes")
        print(classification_report(y_test, y_pred))

        # Generate a confusion matrix that compares the actual test labels (y_test) with the predicted labels (y_pred).
        # It helps visualize how many samples were correctly or incorrectly classified.
        cm = confusion_matrix(y_test, y_pred)

        # Display the confusion matrix in a plot, with labels for both classes (0 and 1).
        # The color map 'Blues' provides a visual representation of the matrix, making it easier to interpret.
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(cmap='Blues')

        # Set the plot's title to "Confusion Matrix" for clarity.
        plt.title("Confusion Matrix")

        # Save and display the confusion matrix plot on the screen.
        fname = Path("..", "figs", "NaiveBayes_ConfusionMatrix.pdf")
        plt.savefig(fname)
        # plt.show()