from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

def find_the_optimal_parameters_using_GridSearch(X, y,
                                                  model,
                                                  param_grid_):
    print("GridSearch")
    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid_,
                               cv=5,
                               scoring='accuracy',
                               n_jobs=-1,
                               verbose=4)
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_model_accuracy = cross_val_score(best_model, X, y, scoring='accuracy').mean()
    print('Accuracy obtained by best model: {:.4f}%'.format(best_model_accuracy * 100))

    return grid_search.best_params_
