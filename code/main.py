from model import StrokeRiskPredictor

def main():
    model = StrokeRiskPredictor("../data/healthcare-dataset-stroke-data.csv")
    X_resampled, y_resampled = model.preprocess_data()
    X_train, X_test, y_train, y_test = model.create_training_and_testing_data(X_resampled, y_resampled)
    model.k_nearest_neighbours_classifier(X_train, X_test, y_train, y_test)
    model.random_forest_classifier(X_train, X_test, y_train, y_test)
    model.logistic_regression_classifier(X_train, X_test, y_train, y_test)
    model.bernoulli_naive_bayes(X_train, X_test, y_train, y_test)
    

if __name__ == "__main__":
    main()