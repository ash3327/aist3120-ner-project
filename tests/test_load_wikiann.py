import torch
from datasets import load_dataset

# Load the CoNLL-2003 dataset
dataset = load_dataset("unimelb-nlp/wikiann", "en", trust_remote_code=True)
dataset.set_format("torch")

# # 'train', 'validation', 'test'.
NER_TAGS = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6}
REV_NER_TAGS = {v:k for k,v in NER_TAGS.items()}

data = dataset['train'][0]
print(data)
print(list(map(REV_NER_TAGS.get, data['ner_tags'].numpy().tolist())))