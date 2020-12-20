import random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_models(X_train, X_test, y_train, y_test):
    random.seed(42)

    lr_classifier = LogisticRegression()
    lr_classifier.fit(X_train, y_train)
    print("Machine is Learning\n")
    score = lr_classifier.score(X_test, y_test)
    print('Accuracy obtained by Logistic Regression: {:.4f}%'.format(score * 100))

    rf_classifier = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    print("Machine is Learning\n")
    rf_classifier.fit(X_train, y_train)
    score = rf_classifier.score(X_test, y_test)
    print('Accuracy obtained by Random Forest Classifier: {:.4f}%'.format(score * 100))

    return (lr_classifier, rf_classifier)