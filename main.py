import csv
import pandas as pd
from preprocessing import preprocess_the_text
from train_models import train_models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

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

   train_models(X_train, X_test, y_train, y_test)
