import sys
import argparse
sys.path.append('.')

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from torch import cuda
from string import punctuation

from eval import Eval
from libs import NER

class MaskedBertNER(NER):
    def __init__(self, model_name="runs/bert_ft_v1", *args, **kwargs):
        self.puncts = punctuation
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)

        device = "cuda:0" if cuda.is_available() else "cpu"
        model = model.to(device)

        # self.nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="max")
        self.nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=device)
    
    def get_entities(self, tokens):
        """
        Compare NER labels between dataset and spaCy.
        
        Args:
            tokens (list): List of tokens/words
            
        Returns:
            list: List of [label, entity string]
        """
        example = ' '.join(tokens)

        def process_text(text):
            return text.replace(' - ', '-')
        
        example = process_text(example)  # Fix for hyphenated words
        for punct in self.puncts:
            example = example.replace(' '+punct, punct)

        ner_results = self.nlp(example)
        # return [(result['entity_group'],process_text(result['word'])) for result in ner_results if result['entity_group'] != 'O']
        # ner_tokens = self.nlp.tokenizer.tokenize(example)
        ner_tokens = [[(v,i,j) for j,v in enumerate(self.nlp.tokenizer.tokenize(token))] for i,token in enumerate(tokens)]
        ner_tokens = [ner_token for sublist in ner_tokens for ner_token in sublist] # flatten
        # subword_to_orig = [orig for (subword, orig) in ner_tokens] #

        # print()
        # print(tokens,'\n\t',ner_results, '\n\t',ner_tokens)

        if len(ner_results) == 0:
            return []
        
        out = list()
        lastid = -1
        lastlabel = None
        lastitem = None

        def insert_entity(cls, entity):
            out.append((cls, entity))

        for result in ner_results:
            id = result['index']-1
            if ner_tokens[id][1] <= lastid or ner_tokens[id][2] != 0:
                continue
            # print('##',result, (result['entity'][2:], tokens[ner_tokens[id][1]]))
            if result['entity'].startswith('B-'):
                if lastitem is not None:
                    insert_entity(lastlabel, lastitem)
                lastlabel = result['entity'][2:]
                lastitem = tokens[ner_tokens[id][1]]
            elif result['entity'].startswith('I-'):
                if lastitem is not None:
                    lastitem += ' ' + tokens[ner_tokens[id][1]]
            # out.append((result['entity'][2:], tokens[ner_tokens[id][1]]))
            lastid = ner_tokens[id][1]
            
        if lastitem is not None:
            insert_entity(lastlabel, lastitem)
        # out.append((lastlabel, lastitem))

        return out
    
"""
Problem:
 - if the head of a word is predicted as B-XXX, the remaining part of the same word is sometimes MISSED.

Example:

V3:
['Portuguesa', '1', 'Atletico', 'Mineiro', '0']
False Positives [('ORG', 'Portug')]
False Negatives [('ORG', 'Portuguesa')]
Predicted entities: [('ORG', 'Portug'), ('ORG', 'Atletico Mineiro')]

['CRICKET', '-', 'LARA', 'ENDURES', 'ANOTHER', 'MISERABLE', 'DAY', '.']
False Positives [('LOC', 'LARA'), ('PER', 'MI')]
False Negatives [('PER', 'LARA')]
Predicted entities: [('LOC', 'LARA'), ('PER', 'MI')]

['Robert', 'Galvin']
False Positives [('PER', 'Robert Gal')]
False Negatives [('PER', 'Robert Galvin')]
Predicted entities: [('PER', 'Robert Gal')]
  7%|████████▊                                                                                                                         | 234/3453 [00:02<00:27, 116.83it/s] 
['Australia', 'gave', 'Brian', 'Lara', 'another', 'reason', 'to', 'be', 'miserable', 'when', 'they', 'beat', 'West', 'Indies', 'by', 'five', 'wickets', 'in', 'the', 'opening', 'World', 'Series', 'limited', 'overs', 'match', 'on', 'Friday', '.']
False Positives [('ORG', 'West Indies')]
False Negatives [('LOC', 'West Indies')]
Predicted entities: [('LOC', 'Australia'), ('PER', 'Brian Lara'), ('ORG', 'West Indies'), ('MISC', 'World Series')]

['All-rounder', 'Greg', 'Blewett', 'steered', 'his', 'side', 'to', 'a', 'comfortable', 'victory', 'with', 'an', 'unbeaten', '57', 'in', '90', 'balls', 'to', 'the', 'delight', 'of', 'the', '42,442', 'crowd', '.']
False Positives [('PER', 'Greg Blew')]
False Negatives [('PER', 'Greg Blewett')]
Predicted entities: [('PER', 'Greg Blew')]

['Man-of-the', 'match', 'Blewett', 'came', 'to', 'the', 'wicket', 'with', 'the', 'total', 'on', '70', 'for', 'two', 'and', 'hit', 'three', 'fours', 'during', 'an', 'untroubled', 'innings', 'lasting', '129', 'minutes', '.']
False Positives [('PER', 'Blew')]
False Negatives [('PER', 'Blewett')]
Predicted entities: [('PER', 'Blew')]

['Lara', 'looked', 'out', 'of', 'touch', 'during', 'his', 'brief', 'stay', 'at', 'the', 'crease', 'before', 'chipping', 'a', 'simple', 'catch', 'to', 'Shane', 'Warne', 'at', 'mid-wicket', '.']
False Positives [('PER', 'Shane War')]
False Negatives [('PER', 'Shane Warne')]
Predicted entities: [('PER', 'Lara'), ('PER', 'Shane War')]

V4:

"""

def main():
    parser = argparse.ArgumentParser(description="Masked BERT NER Evaluation")
    parser.add_argument("--model_name", type=str, default="runs/bert_ft_v1", help="Path to the model")
    args = parser.parse_args()

    # Create NER comparison object
    ner_compare = MaskedBertNER(model_name=args.model_name)

    # print(ner_compare.get_entities(['Robert', 'Galvin']))
    # print(ner_compare.get_entities(['CRICKET', '-', 'SHEFFIELD', 'SHIELD', 'SCORE', '.']))
    # print(ner_compare.get_entities(['Pau-Orthez', '(', 'France', ')', '9', '5', '4', '14']))
    # print(ner_compare.get_entities(['10.', 'Troy', 'Benson', '(', 'U.S.', ')', '22.56']))
    # print(ner_compare.get_entities(['Lara', 'looked', 'out', 'of', 'touch', 'during', 'his', 'brief', 'stay', 'at', 'the', 'crease', 'before', 'chipping', 'a', 'simple', 'catch', 'to', 'Shane', 'Warne', 'at', 'mid-wicket', '.']))
    Eval.evaluate_dataset(ner_compare, dataset="conll", split="test")
    Eval.evaluate_dataset(ner_compare, dataset="wikiann", split="test")

if __name__ == "__main__":
    main()