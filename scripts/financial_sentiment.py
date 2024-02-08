import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("../data/financial_sentiment.csv",
                 encoding='cp437',
                 header=None,
                 names=['sentiment', 'text'])

# processing
le = LabelEncoder()
df['y'] = le.fit_transform(df['sentiment'])
df.head()
df['sentiment'].hist()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.1, stratify=df['y'])
print(train_df['sentiment'].value_counts()/len(train_df))

# term frequency vecotrizer
tf_vectorizer = CountVectorizer(max_df=0.99, min_df=2,
                                lowercase=True,
                                stop_words='english')

# sparse matrix of vectorized sentences
train_X = tf_vectorizer.fit_transform(train_df['text'].values)
test_X = tf_vectorizer.transform(test_df['text'].values)
print(tf_vectorizer.get_feature_names())

# model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(multi_class='multinomial')
model.fit(train_X, train_df['y'])

test_preds = model.predict(test_X)
acc = accuracy_score(test_df['y'], test_preds)
print("model test accuracy without further processing: {:.3f}".format(acc))

# stemming to improve model
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


print(df['text'][0])
ps = PorterStemmer()
print(" ".join([ps.stem(word) for word in word_tokenize(df['text'][0])]))

def stem_sentence(s):
    return " ".join([ps.stem(word) for word in word_tokenize(s)])

train_df['processed_text'] = train_df['text'].map(stem_sentence)
test_df['processed_text'] = test_df['text'].map(stem_sentence)

# term frequency vecotrizer
tf_vectorizer = CountVectorizer(max_df=0.99, min_df=2,
                                lowercase=True,
                                stop_words='english')
# sparse matrix of vectorized sentences
train_X = tf_vectorizer.fit_transform(train_df['processed_text'].values)
test_X = tf_vectorizer.transform(test_df['processed_text'].values)
print(tf_vectorizer.get_feature_names())

model = LogisticRegression(multi_class='multinomial')
model.fit(train_X, train_df['y'])

test_preds = model.predict(test_X)
acc = accuracy_score(test_df['y'], test_preds)
print("model test accuracy with stemming: {:.3f}".format(acc))


# include ngrams for more context
tf_vectorizer = CountVectorizer(max_df=0.99, min_df=5,
                                lowercase=True,
                                stop_words='english',
                                ngram_range=(1, 2))

train_X = tf_vectorizer.fit_transform(train_df['processed_text'].values)
test_X = tf_vectorizer.transform(test_df['processed_text'].values)
print(tf_vectorizer.get_feature_names())
model = LogisticRegression(multi_class='multinomial')
model.fit(train_X, train_df['y'])

test_preds = model.predict(test_X)
acc = accuracy_score(test_df['y'], test_preds)
print("model test accuracy with ngrams: {:.3f}".format(acc))

# get information on decisive words for sentiment
# shape [3, ncoeffs] for logistic coefficients per class label (neutral, pos, neg)
coeffs = (-model.coef_).argsort(axis=-1)[:, :10]
print(coeffs)
words = tf_vectorizer.get_feature_names()
for i, idx in enumerate(coeffs):
    label = le.inverse_transform([i])
    print(label)
    print([words[i] for i in idx])
    print("="*10)

from sklearn.metrics import classification_report
print(classification_report(test_df['y'], test_preds))


# --------------------------------------------------------- SPACY ------------------------------------------------------------------
# use spacy for NLP and named entities
# get spacy model via $python -m spacy download en_core_web_sm'
# load with spacy.load('en_core_web_sm')

import spacy
nlp = spacy.load('en_core_web_sm')

doc = nlp(df['text'].values[0])
print(doc)

for entity in doc.ents:
    print(entity.text, entity.label_)

df['ents'] = df['text'].map(lambda x: [(entity.text, entity.label_) for entity in nlp(x).ents])
df['ent_types'] = df['ents'].map(lambda x: set([ent[1] for ent in x]))

# format text to replace all the individual entities with their entity type
def replace_entities(text, entities):
    for (entity, ent_type) in entities:
        text = text.replace(entity, ent_type)
    return text

df['format_text'] = df.apply(lambda x: replace_entities(x["text"], x["ents"]), axis=1)

train_df, test_df = train_test_split(df, test_size=0.1, stratify=df['y'])
print(train_df['sentiment'].value_counts()/len(train_df))
 # stemming
train_df['processed_text'] = train_df['format_text'].map(stem_sentence)
test_df['processed_text'] = test_df['format_text'].map(stem_sentence)

# term frequency vecotrizer
tf_vectorizer = TfidfVectorizer(max_df=0.99, min_df=5,
                                lowercase=True,
                                stop_words='english')

# sparse matrix of vectorized sentences
train_X = tf_vectorizer.fit_transform(train_df['processed_text'].values)
test_X = tf_vectorizer.transform(test_df['processed_text'].values)
print(tf_vectorizer.get_feature_names())

# model
model = LogisticRegression(multi_class='multinomial')
model.fit(train_X, train_df['y'])

test_preds = model.predict(test_X)
acc = accuracy_score(test_df['y'], test_preds)
print("model test accuracy without further processing: {:.3f}".format(acc))

print('finished')