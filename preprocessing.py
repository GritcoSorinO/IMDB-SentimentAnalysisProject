import re
from bs4 import BeautifulSoup
from gensim.parsing.preprocessing import remove_stopwords


def eliminate_html_tags(review_text):
    soup = BeautifulSoup(review_text, 'html.parser')
    return soup.get_text(separator=' ')


def eliminate_special_characters(review_text):
    pattern = r'[^a-zA-Z\s]'
    return re.sub(pattern, ' ', review_text)


def preprocess_the_text(review_text):
    review_text = eliminate_html_tags(review_text)
    review_text = eliminate_special_characters(review_text)
    review_text = review_text.lower()
    review_text = remove_stopwords(review_text)
    return review_text
