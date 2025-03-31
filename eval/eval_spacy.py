import sys
sys.path.append('.')

from datasets import load_dataset
import random
from tqdm import tqdm

from spacy_ner import SpacyNER
from eval import Eval

def main(dataset="conll", split="test"):
    # Create NER comparison object
    ignore_types = ['DATE', 'TIME', 'CARDINAL', 'ORDINAL', 'MONEY', 'PERCENT', 'QUANTITY', 'PRODUCT', 'WORK_OF_ART']
    ner_compare = SpacyNER(ignore_types=ignore_types)

    Eval.evaluate_dataset(ner_compare, dataset, split)

if __name__ == "__main__":
    main(dataset="conll", split="test")
    main(dataset="conll", split="validation")
    main(dataset="conll", split="train")
    main(dataset="wikiann", split="test")
    main(dataset="wikiann", split="validation")
    main(dataset="wikiann", split="train")