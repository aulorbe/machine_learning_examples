from nltk.corpus import brown
from nltk import bigrams
from collections import Counter, defaultdict

# Create bigram language model with add-one smoothing
# Lowercase words

# Test language model on:
    # Real sentence v. fake sentence
    # Real sentence not in training corpus

# Take the log probability of your bigrams
# Normalize each sentences


# From https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-language-model-nlp-python-code/:
# Placeholder model
model = defaultdict(lambda: defaultdict(lambda: 0))

# Write out Markov assumption
for sentence in brown.sents():
    for w1, w2 in bigrams(sentence):
        model[w1][w2] += 1

# # Probabilities
for w1 in model:
    total_count = float(sum(model[w1].values()))
    # print(total_count)
    for w2 in model[w1]:
        model[w1][w2] /= total_count


# From course exercise overview:
# Should've used brown.py from rnn_class directory


