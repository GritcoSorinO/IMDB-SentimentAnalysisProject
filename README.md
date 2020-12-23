# IMDB-SentimentAnalysisProject

The labeled data set consists of 25,000 IMDB movie reviews, specially selected for sentiment analysis. The sentiment of reviews is binary, meaning the IMDB rating < 5 results in a sentiment score of 0, and rating >=7 have a sentiment score of 1. (Dataset is in attachment labeledTrainData.tsv)


Data fields:

id - Unique ID of each review;

sentiment - Sentiment of the review; 1 for positive reviews and 0 for negative reviews;

review - Text of the review.

Requirements:
- python 3.7, 3.8
- PyCharm IDE 2020 
- Anaconda 3
- JupyterNotebook


Libraries needed to be installed:
- bs4 v.0.0.1
- gensim v.3.8.3
- numpy v.1.18.0
- pandas v.1.15
- regex v.2020.11.13
- scikit-learn 0.23.2

Observation: 
File IMDBSentimentAnalysisProject.ipynb is a JupyterNotebook, 
if you want to see how the code ran and some explanatory text open it.
