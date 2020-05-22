import re
import unidecode
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from bs4 import BeautifulSoup


def clean_data(text):
    soup = BeautifulSoup(text)
    html_pattern = re.compile('<.*?>')
    text = html_pattern.sub(r' ', soup.text)
    text = unidecode.unidecode(text)
    text = re.sub('[^A-Za-z0-9.]+', ' ', text)
    text = text.lower()
    
    return text
    
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train',
                                  categories=categories,
                                  shuffle=True,
                                  remove=('headers', 'footers', 'quotes'),
                                  random_state=42)
print(twenty_train.target_names)

data = twenty_train.data
data = [clean_data(txt) for txt in data]
target = twenty_train.target

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(data)
print(X_train_counts.shape)

mnb = MultinomialNB()
print("Mean Accuracy: {:.2}".format(cross_val_score(mnb, X_train_counts, target, cv=5).mean()))
# We get 84% accuracy here, during my experiment
