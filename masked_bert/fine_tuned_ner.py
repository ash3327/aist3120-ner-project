import sys
sys.path.append('.')

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from string import punctuation
from torch import cuda

from eval import Eval
from libs import NER

class BertNER(NER):
    def __init__(self, model_name="dslim/bert-large-NER", aggregation_strategy="max", *args, **kwargs):
        self.puncts = punctuation
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)

        device = "cuda:0" if cuda.is_available() else "cpu"
        model = model.to(device)

        self.nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy=aggregation_strategy, device=device)
    
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
        return [(result['entity_group'],process_text(result['word'])) for result in ner_results if result['entity_group'] != 'O']
    
def main(dataset="conll", split="test", model_name="dslim/bert-large-NER", aggregation_strategy="max"):
    # Create NER comparison object
    ner_compare = BertNER(model_name=model_name, aggregation_strategy=aggregation_strategy)

    Eval.evaluate_dataset(ner_compare, dataset, split)

if __name__ == "__main__":
    main(dataset="conll", split="test")