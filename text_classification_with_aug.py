import re
import unidecode
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from bs4 import BeautifulSoup
import nltk
from nltk.tag import pos_tag
from nltk import sent_tokenize
from nltk.corpus import wordnet

def get_synonym_for_word(word):
    """returns the synonym given word if found, otherwise returns the same word"""
    
    synonyms = []
    for syn in wordnet.synsets(word):

        for l in syn.lemmas():
            synonyms.append(l.name())
    synonyms = [syn for syn in synonyms if syn!=word]
    if len(synonyms) == 0:
        return word
    else:
        return synonyms[0]

def augment_sentence_wordnet(sentence, filters=['NN', 'JJ']):
    """Augments words in sentence which are filtered by pos tags"""
    
    pos_sent = pos_tag(sentence.split())
    new_sent = []
    for word,tag in pos_sent:
        if tag in filters:
            new_sent.append(get_synonym_for_word(word))
        else:
            new_sent.append(word)
            
    return " ".join(new_sent)

def augment_data(data, target):
    """Creates augmented data using wordnet synonym imputation."""
    
    aug_data = []
    aug_target = []
    for row, t in zip(data, target):
        aug_row = []
        row_sents = sent_tokenize(row)
        #print("row_sents", row_sents)
        for line in row_sents:
            line = augment_sentence_wordnet(line)
            aug_row.append(line)
        row_sents = " ".join(aug_row)
        
        #print(row_sents)
        aug_data.append(row)
        aug_data.append(row_sents)
        aug_target.append(t)
        aug_target.append(t)
        #print(len(aug_data))
    return aug_data, aug_target

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

aug_data, aug_target = augment_data(data, target)

count_vect = CountVectorizer()
X_train_aug_counts = count_vect.fit_transform(aug_data)
print(X_train_aug_counts.shape)

mnb_aug = MultinomialNB()
print("Mean Accuracy: {:.2}".format(cross_val_score(mnb_aug, X_train_aug_counts, aug_target, cv=5).mean()))
# I get 85% accuracy here, during my experiment. 
