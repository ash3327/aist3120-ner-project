import sys
sys.path.append('.')

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from string import punctuation

from eval import Eval
from libs import NER

class BertNER(NER):
    def __init__(self, model_name="dslim/bert-large-NER", *args, **kwargs):
        self.puncts = punctuation
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

        self.nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    
    def get_entities(self, tokens):
        """
        Compare NER labels between dataset and spaCy.
        
        Args:
            tokens (list): List of tokens/words
            
        Returns:
            list: List of [label, entity string]
        """
        example = ' '.join(tokens)
        for punct in self.puncts:
            example = example.replace(' '+punct, punct)

        ner_results = self.nlp(example)
        # print(ner_results)
        # token_ids = tokenizer(example)['input_ids']
        # tokens = tokenizer.convert_ids_to_tokens(token_ids)
        # print(tokens)
        if len(ner_results) == 0:
            return []
        
        out = list()
        lastid = ner_results[0]['index']
        lastitem = ner_results[0]['word']
        lastlabel = ner_results[0]['entity'][2:]

        for result in ner_results[1:]:
            if result['index'] == lastid+1:
                if result['word'].startswith('##'):
                    lastitem += result['word'][2:]
                elif result['word'] in self.puncts:
                    lastitem += result['word']
                else:
                    lastitem += ' ' + result['word']
                lastid += 1
            else:
                if lastitem:
                    out.append((lastlabel, lastitem))
                if result['word'].startswith('##'):
                    lastid = -1
                    lastitem = None
                    lastlabel = None
                else:
                    lastid = result['index']
                    lastitem = result['word']
                    lastlabel = result['entity'][2:]
        out.append((lastlabel, lastitem))

        return out
    
def main(dataset="conll", split="test"):
    # Create NER comparison object
    ner_compare = BertNER()

    Eval.evaluate_dataset(ner_compare, dataset, split)

if __name__ == "__main__":
    main(dataset="conll", split="test")