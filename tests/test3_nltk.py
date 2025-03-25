# https://www.ibm.com/think/topics/named-entity-recognition
# https://www.nltk.org/
# https://medium.com/data-science/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from nltk.chunk import conlltags2tree, tree2conlltags
from nltk import ne_chunk
from pprint import pprint

nltk.download('words')
nltk.download('punkt_tab')
nltk.download('maxent_ne_chunker_tab')
nltk.download('averaged_perceptron_tagger_eng')

# sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good."""
# tokens = nltk.word_tokenize(sentence)
# print(tokens)

# tagged = nltk.pos_tag(tokens)
# print(tagged)

ex = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

sent = preprocess(ex)
print(sent)

# custom chunk pattern
pattern = 'NP: {<DT>?<JJ>*<NN>}'
cp = nltk.RegexpParser(pattern)
cs = cp.parse(sent)
print(cs)

# IOB tags format
iob_tagged = tree2conlltags(cs)
pprint(iob_tagged)

ne_tree = ne_chunk(pos_tag(word_tokenize(ex)))
print(ne_tree)