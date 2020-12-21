from obtain_data_frame_from_file_path import obtain_data_frame_from_file_path
from preprocessing import preprocess_the_text
from train_models import train_models
from find_the_optimal_parameters import find_the_optimal_parameters_using_GridSearch
from apply_cross_validation_to_models import apply_cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
   file_path = 'data/labeledTrainData.tsv'
   df = obtain_data_frame_from_file_path(file_path)

   df['review'] = df['review'].apply(preprocess_the_text)

   #1. Vectorize review texts
   vectorizer = CountVectorizer(min_df=0, lowercase=False)
   vectorizer.fit(df['review'])

   reviews = df['review'].values
   sentiments = df['sentiment'].values

   #2. Train 2 models Logistic Regression and Random Forest
   test_size_ = 0.2
   random_seed = 42

   reviews_train, review_test, y_train, y_test = train_test_split(reviews,
                                                                  sentiments,
                                                                  test_size=test_size_,
                                                                  random_state = random_seed)
   X_train = vectorizer.transform(reviews_train)
   X_test = vectorizer.transform(review_test)

   rf_model, lr_model = train_models(X_train, X_test,
                                     y_train, y_test,
                                     random_seed)

   X = vectorizer.transform(reviews)
   y = sentiments

   '''
      3. Compare the value of Accuracy metric accross 
         two models using cross-validation.
   '''

   lr_model = LogisticRegression(random_state=random_seed)

   print("K-Fold cross-validation for Logistic Regression model.")
   apply_cross_validation(X, y, lr_model)

   rf_model = RandomForestClassifier(random_state=random_seed)

   print("K-Fold cross-validation for Random Forest model.")
   apply_cross_validation(X, y, rf_model)

   '''
      4. Find the optimal parameters for the model 
         that performs better using GridSearch
   '''

   rf_model = RandomForestClassifier(random_state=random_seed)
   param_grid = {
      'n_estimators': [100, 200, 500, 750, 1000],
      'max_features': ['auto', 'sqrt', 'log2'],
      'max_depth': [8, 10, 15, 20],
      'criterion': ['gini', 'entropy']
   }

   print(find_the_optimal_parameters_using_GridSearch(X, y, rf_model, param_grid))

   lr_model = LogisticRegression(random_state=random_seed)
   param_grid = {
      'penalty': ['l2', 'none'],
      'tol': [1e-4, 1e-5, 1e-6],
      'C': [0.01, 0.1, 1.0, 10],
      'max_iter': [100, 200, 500]
   }

   print(find_the_optimal_parameters_using_GridSearch(X, y, lr_model, param_grid))