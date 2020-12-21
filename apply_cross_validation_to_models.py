from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def apply_cross_validation(X, y, model):

    cv_ = KFold(n_splits=5, random_state=42, shuffle=True)
    scores = cross_val_score(model,
                             X, y,
                             scoring='accuracy',
                             cv=cv_,
                             n_jobs=-1)

    print(scores)
    print('Accuracy obtained by Random Forest Classifier: {:.2f}%, ({:.4f})'.format(mean(scores)*100, std(scores)))

    return mean(scores)