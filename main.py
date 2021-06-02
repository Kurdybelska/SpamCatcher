from collections import Counter

import sklearn.model_selection
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
import pandas as pd

data_set = []
classes = ['spam', 'ham']

tokenizer = RegexpTokenizer(r'\w+')
snowball = SnowballStemmer(language='english')
# Source: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
with open('SMSSpamCollection', 'r', encoding="utf8") as file:
    for line in file:
        label, msg = line.rstrip().split('\t', 2)
        #print(msg)
        tokenized_msg = tokenizer.tokenize(msg)
        #print(tokenized_msg)
        stemmed_msg = [snowball.stem(token) for token in tokenized_msg]
        #print(stemmed_msg)
        data_set.append((label, stemmed_msg))

train_set, test_set = sklearn.model_selection.train_test_split(data_set, train_size=0.8)
print(f'Train len: {len(train_set)}. Test len: {len(test_set)}')

train_spam = []
train_ham = []
for sms in train_set :
    msg = sms[1]
    label = sms[0]
    if label == "spam":
        train_spam.append(msg)
    else :
        train_ham.append(msg)


vocab_spam = []
for sms in train_spam :
    for word in sms:
        vocab_spam.append(word)

vocab_spam = list(dict.fromkeys(vocab_spam))

vocab_ham = []
for sms in train_ham :
    for word in sms:
        vocab_ham.append(word)

vocab_ham = list(dict.fromkeys(vocab_ham))


dict_spamicity = {}
for word in vocab_spam :
    msgs = 0
    for sms in train_spam :
        if word in sms :
            msgs += 1
    total_spam = len(train_spam)
    spamicity = (msgs+1) / (total_spam+2)
    dict_spamicity[word] = spamicity

dict_hamicity = {}
for word in vocab_ham :
    msgs = 0
    for sms in train_ham :
        if word in sms :
            msgs += 1
    total_ham = len(train_ham)
    hamicity = (msgs+1) / (total_ham+2)
    dict_hamicity[word] = hamicity


prob_spam = len(train_spam) / (len(train_spam)+(len(train_ham)))
prob_ham = len(train_ham) / (len(train_spam)+(len(train_ham)))

spam_counter = 0
ham_counter = 0
poprawne = 0

for sms in test_set :
    msg = sms[1]
    pr_spam = prob_spam
    pr_ham = prob_ham
    for word in msg :
        if word in dict_spamicity :
            pr_spam *= dict_spamicity[word]
        elif word in dict_hamicity :
            pr_ham *= dict_hamicity[word]
    if pr_spam > pr_ham :
        spam_counter += 1
        if sms[0] == "spam" :
            poprawne += 1
    else :
        ham_counter += 1
        if sms[0] == "ham" :
            poprawne += 1


print("ham: ",ham_counter)
print("spam: ",spam_counter)
poprawne_procent = (poprawne*100) / len(test_set)
print(" procent dobrze rozpoznanych: ",poprawne_procent,"%")




