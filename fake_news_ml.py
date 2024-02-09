# %% 1
import itertools
import time
import pandas as pd
import numpy as np
import warnings
from langdetect import detect
import contractions
import nltk
from matplotlib import pyplot as plt, pyplot
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')
np.random.seed = 42

df = pd.read_csv('PreProcessedData.csv')

# delete id column
df = df.drop(df.columns[0], axis=1)

# change of types
df["Ground Label"] = np.where(df["Ground Label"] == "true", 1, 0)
df['text'] = df['text'].astype('string')
df['title'] = df['title'].astype('string')

# info before preprocessing
na_ratio_cols = df.isna().mean(axis=0)
print(na_ratio_cols)

df.info()
df.hist(column='Ground Label', bins = 2)


# %% 2

# count the number of occurrences, and then remove duplicates - this way we do not lose information about the number of occurrences

df['occurrences'] = df.groupby(['title', 'text'])['title'].transform('count')
df = df.drop_duplicates()

# detect language - because all non-English are fake (767/59445) - all to be removed
def is_english(text):
    try:
        lang = detect(text)
    except:
        return False
    return lang == 'en'

# remove non-English
df['is_english'] = df['text'].apply(lambda x: is_english(x))
df = df[df['is_english']]
df = df.drop(['is_english'], axis=1)


# %% 3

# data split

X = df.drop('Ground Label', axis=1)
y = df['Ground Label']

df, df_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

df, X_val, y_train, y_val = train_test_split(
    df, y_train, stratify=y_train, test_size=0.3, random_state=42
)


# df, y_train - training set
# df_test, y_test - internal validation set
# X_val, y_val - external validation set

# %% 4

# fill NA values with an empty string or the number 1

def changeNA(text):
    if not isinstance(text, str) or pd.isnull(text) or len(text) < 3:
        return ""
    return text

def nan_to_one(x):
    if pd.isna(x):
        return 1
    return x

df.isnull().any()

df['title'] = df['title'].apply(lambda x: changeNA(x))
df['text'] = df['text'].apply(lambda x: changeNA(x))

df_test['title'] = df_test['title'].apply(lambda x: changeNA(x))
df_test['text'] = df_test['text'].apply(lambda x: changeNA(x))

df['occurrences'] = df['occurrences'].apply(lambda x: nan_to_one(x))
df_test['occurrences'] = df_test['occurrences'].apply(lambda x: nan_to_one(x))

df.info()


# contractions
# changes shortcuts to full words (e.g.: u - you, don't - do not)
def remove_contractions(text):
    try:
        s = ' '.join([contractions.fix(word) for word in text.split()])
    except:
        s = text
    return s


df['title'] = df['title'].apply(lambda x: remove_contractions(x))
df['text'] = df['text'].apply(lambda x: remove_contractions(x))

df_test['title'] = df_test['title'].apply(lambda x: remove_contractions(x))
df_test['text'] = df_test['text'].apply(lambda x: remove_contractions(x))

print('contractions removed')

# %% 5
# stop words
# removes words that do not carry information by themselves, e.g. the sentence "Donald Trump is being under control of police" will be changed to "Donald Trump under control police"
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return " ".join([word for word in str(text).lower().split() if word not in stop_words])


df['title'] = df['title'].apply(lambda x: remove_stopwords(x))
df['text'] = df['text'].apply(lambda x: remove_stopwords(x))

df_test['title'] = df_test['title'].apply(lambda x: remove_stopwords(x))
df_test['text'] = df_test['text'].apply(lambda x: remove_stopwords(x))

print('stopwords removed')


# stemming
# changes words to their base form without endings

stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])


df['title'] = df['title'].apply(lambda x: stem_words(x))
df["text"] = df["text"].apply(lambda x: stem_words(x))

df_test['title'] = df_test['title'].apply(lambda x: stem_words(x))
df_test["text"] = df_test["text"].apply(lambda x: stem_words(x))

print("stemming done")


# tokenization
# We create a new column with an array of words used in the text, to create columns counting the number of proper nouns etc.

df['tokenized_text'] = df['text'].apply(lambda x: word_tokenize(x))
df['tokenized_title'] = df['title'].apply(lambda x: word_tokenize(x))

df_test['tokenized_text'] = df_test['text'].apply(lambda x: word_tokenize(x))
df_test['tokenized_title'] = df_test['title'].apply(lambda x: word_tokenize(x))

print("tokenization done")


def count_proper_nouns(token_text):
    proper_nouns_counter = 0
    tagged = nltk.pos_tag(token_text)
    for i in range(len(tagged)):
        word, pos = tagged[i]
        if pos == 'NNP':
            if i != 0 and not tagged[i - 1][1] in ['.', '!', '?']:
                proper_nouns_counter += 1
    return proper_nouns_counter


# Counting proper nouns, number of words and punctuation marks (in relation to the length of the text)
def word_counting(df):
    df['words_counter'] = df['text'].apply(lambda x: len(x.split()))
    df['proper_nouns_counter'] = df['tokenized_text'].apply(lambda x: count_proper_nouns(x)) / df['words_counter']
    df['coma_counter'] = df['text'].apply(lambda x: x.count(',')) / df['words_counter']
    df['exclamation_mark_counter'] = df['text'].apply(lambda x: x.count('!')) / df['words_counter']
    df['question_mark_counter'] = df['text'].apply(lambda x: x.count('?')) / df['words_counter']

word_counting(df)
word_counting(df_test)

print("word counting done")

# %% 6

df.hist(column='words_counter', bins = 25)
df.hist(column='proper_nouns_counter', bins = 25)
df.hist(column='coma_counter', bins = 25)
df.hist(column='exclamation_mark_counter', bins = 25)
df.hist(column='question_mark_counter', bins = 25)
df.hist(column='occurrences')
corrMatrix = df.corr()
plt.show()

'''
almost nothing is removed
def correlations(df, max_acceptable_corr):
        corr = df.corr()
        correlations_map = {}
        for col in corr.columns: 
            correlations_map[col] = corr.loc[(abs(corr[col]) > max_acceptable_corr).tolist(), col].index.tolist()
        keys_to_remove = []
        for key in correlations_map:
            if len(correlations_map[key]) <= 1:
                keys_to_remove.append(key)
            else:
                correlations_map[key].remove(key)
        for key in keys_to_remove:
            del correlations_map[key]
        return correlations_map


def remove_correlated_cols(df, max_acceptable_corr):
    start_cols = df.shape[1]
    c = correlations(df, max_acceptable_corr)
    for key in c:
        if not key in df.columns:
            continue
        cols = c[key]
        for col in cols:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
    print("There are " + str(df.shape[1]) + " columns left.")
    print("Removed " + str(start_cols - df.shape[1]) + " columns.")

remove_correlated_cols(vec_title, 0.8) 
'''


# %% 7

# vectorization - text as occurrence columns, because the computer doesn't understand sentences, it just figures out where which words were used with which other words

# we set a minimum and maximum frequency threshold, according to which we will take only portion of the words to the model

# for max_df = 0.7 and/or min_df = 0.005, the predictability scores were worse
max_df = 0.6
min_df = 0.01

vectorizerTitle = TfidfVectorizer(max_df=max_df, min_df=min_df)
vectorizerText = TfidfVectorizer(max_df=max_df, min_df=min_df)

vectorizerTitle.fit(df['title'])
vectorizerText.fit(df['text'])

cols_title = vectorizerTitle.get_feature_names()
cols_text = vectorizerText.get_feature_names()
vec_title = pd.DataFrame.sparse.from_spmatrix(vectorizerTitle.transform(df['title']), columns=cols_title)
vec_text = pd.DataFrame.sparse.from_spmatrix(vectorizerText.transform(df['text']), columns=cols_text)

vec_title_test = pd.DataFrame.sparse.from_spmatrix(vectorizerTitle.transform(df_test['title']), columns=cols_title)
vec_text_test = pd.DataFrame.sparse.from_spmatrix(vectorizerText.transform(df_test['text']), columns=cols_text)

# merging
df = df.reset_index(drop=True)
df_numeric = df.iloc[:, [2, 5, 6, 7, 8, 9]]
X_train = pd.concat([df_numeric, vec_text, vec_title], axis=1)


df_test = df_test.reset_index(drop=True)
df_test_num = df_test.iloc[:, [2, 5, 6, 7, 8, 9]]
X_test = pd.concat([df_test_num, vec_text_test, vec_title_test], axis=1, join='inner')

print("vectorization done")

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

X_train.to_csv("train_preprocessed_full.csv", index= False)
X_test.to_csv("test_preprocessed_full.csv", index= False)
y_train.to_csv("y_train_preprocessed_full.csv", index= False)
y_test.to_csv("y_test_preprocessed_full.csv", index= False)
#
#%% 8
#
X_train = pd.read_csv('train_preprocessed_full.csv')
X_test = pd.read_csv('test_preprocessed_full.csv')
y_train = np.ravel(pd.read_csv("y_train_preprocessed_full.csv"))
y_test = np.ravel(pd.read_csv("y_test_preprocessed_full.csv"))

# X_train.columns = X_train.columns.astype(str)
# X_test.columns = X_test.columns.astype(str)

# let's see how model will turn out excluding word 'reuters' (often appearing when news is true)
X_train = X_train.drop('reuters', axis=1)
X_test = X_test.drop('reuters', axis = 1)

classifiers = [
    ExtraTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    GradientBoostingClassifier(),
    DecisionTreeClassifier(),
    XGBClassifier(),
    MultinomialNB(),
]


def important_features_plot(features, importance, name):
    sorted_feats = ([x for _, x in sorted(zip(importance, features), reverse=True)])[0:24]
    importance_vals = sorted(importance, reverse=True)[0:24]
    # creating the bar plot
    plt.bar(sorted_feats, importance_vals)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title(name)
    plt.xticks(rotation=75)
    pyplot.tight_layout()
    plt.show()

columns = list(X_train.columns)
models_df = pd.DataFrame()
for model in classifiers:
    # creation of models, timing and checking their scores
    name = model.__class__.__name__

    start_time = time.time()

    model_result = model.fit(X_train, y_train)
    y_hat = model_result.predict(X_test)
    y_probs = model_result.predict_proba(X_test)[:, 1]

    end_time = time.time()

    # lets check the results
    presicision = precision_score(y_test, y_hat)
    accuracy = accuracy_score(y_test, y_hat)
    recall = recall_score(y_test, y_hat)
    f1 = f1_score(y_test, y_hat)
    roc_auc = roc_auc_score(y_test, y_probs)

    if name=="XGBClassifier":
        plot_importance(model, max_num_features=25)
        pyplot.tight_layout()
        plt.show()
    else:
        print(name)
        if name=="LogisticRegression" or name=="MultinomialNB":
            importances = list(model.coef_[0])
        else:
            importances = list(model.feature_importances_)
        print("Klasyfikator: " + name)
        print("Kolumny o wpływie o wartości 0: " + str(importances.count(0)))
        print("Najbardziej znacząca kolumna: " + columns[importances.index(max(importances))])
        important_features_plot(columns, importances, name)


    # collecting results
    param_dict = {
        'model': name,
        'precision': presicision,
        'accuracy': accuracy,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'time_elapsed': end_time - start_time
    }

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    pyplot.plot(fpr, tpr, linestyle='-', label=model.__class__.__name__)
    pyplot.show()

    models_df = models_df.append(pd.DataFrame(param_dict, index=[0]))

print('basic modeling done')

# let's take the best ones and play with the voting with their combinations, maybe it will be better

def voting_cassifier(voting, classifiers):

    name = voting

    estimators=[]
    for clf in classifiers:
        clf_name = ''.join(c for c in clf.__class__.__name__ if c.isupper())
        estimators.append((clf_name, clf))
        name += ' ' + clf_name
    vc = VotingClassifier(estimators=estimators, voting=voting)

    start_time = time.time()
    model = vc.fit(X_train, y_train)

    y_hat = model.predict(X_test)
    y_probs = model_result.predict_proba(X_test)[:, 1]
    end_time = time.time()

    presicision = precision_score(y_test, y_hat)
    accuracy = accuracy_score(y_test, y_hat)
    recall = recall_score(y_test, y_hat)
    f1 = f1_score(y_test, y_hat)
    roc_auc = roc_auc_score(y_test, y_probs)


    # collecting results

    param_dict = {
        'model': name,
        'precision': presicision,
        'accuracy': accuracy,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'time_elapsed': end_time - start_time
    }

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    pyplot.plot(fpr, tpr, linestyle='--', label='_nolegend_')

    return param_dict


best_classifiers = [GradientBoostingClassifier(),
                       RandomForestClassifier(),
                       LogisticRegression(),
                       XGBClassifier()]

for L in range(2, len(best_classifiers) + 1):
    for subset in itertools.combinations(best_classifiers, L):

        vcs = voting_cassifier('soft', list(subset))
        models_df = models_df.append(pd.DataFrame(vcs, index=[0]))
        if len(subset)>2:
            vch = voting_cassifier('hard', list(subset))
            models_df = models_df.append(pd.DataFrame(vch, index=[0]))


pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()

print('modeling done')
models_df.to_csv('models-full.csv', index=False)