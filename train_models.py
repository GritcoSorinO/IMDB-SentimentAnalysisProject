import random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_models(X_train, X_test, y_train, y_test, random_seed):

    lr_classifier = LogisticRegression(random_state=random_seed)

    print("Train Logistic Regression model.")
    lr_classifier.fit(X_train, y_train)
    score = lr_classifier.score(X_test, y_test)
    print('Accuracy obtained by Logistic Regression: {:.2f}%'.format(score * 100))

    rf_classifier = RandomForestClassifier(n_estimators=200,
                                           n_jobs=-1,
                                           random_state=random_seed)
    print("\nTrain Random Forest Classifier model.")
    rf_classifier.fit(X_train, y_train)
    score = rf_classifier.score(X_test, y_test)
    print('Accuracy obtained by Random Forest Classifier: {:.2f}%'.format(score * 100))

    return (lr_classifier, rf_classifier)