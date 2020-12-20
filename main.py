import csv
import pandas as pd
from preprocessing import preprocess_the_text
from train_models import train_models
from find_the_optimal_parameters import find_the_optimal_parameters_using_GridSearch
from apply_cross_validation_to_models import apply_cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
   filepath = 'data/labeledTrainData.tsv'
   tsv_file = open(filepath, encoding='utf-8')
   read_tsv = csv.reader(tsv_file, delimiter='\t')
   df_list = list()

   for row in read_tsv:
      df_list.append(row)
   df = pd.DataFrame(df_list[1:], columns = df_list[0])

   df['review'] = df['review'].apply(preprocess_the_text)

   vectorizer = CountVectorizer(min_df=0, lowercase=False)
   vectorizer.fit(df['review'])

   reviews = df['review'].values
   sentiments = df['sentiment'].values

   test_size_ = 0.2
   random_seed = 42

   reviews_train, review_test, y_train, y_test = train_test_split(reviews,
                                                                  sentiments,
                                                                  test_size=test_size_,
                                                                  random_state = random_seed)
   X_train = vectorizer.transform(reviews_train)
   X_test = vectorizer.transform(review_test)

   #rf_model, lr_model = train_models(X_train, X_test, y_train, y_test)

   X = vectorizer.transform(reviews)
   y = sentiments

   rf_model = RandomForestClassifier(random_state=random_seed)
   param_grid = {
      'n_estimators': [100, 200, 500, 750, 1000],
      'max_features': ['auto', 'sqrt', 'log2'],
      'max_depth': [8, 10, 15, 20],
      'criterion': ['gini', 'entropy']
   }

   '''
   print("KFold LogReg")
   apply_cross_validation(X, y, lr_model)

   print("KFold RandomForest")
   apply_cross_validation(X, y, rf_model)
   '''
   print(find_the_optimal_parameters_using_GridSearch(X, y, rf_model, param_grid))

