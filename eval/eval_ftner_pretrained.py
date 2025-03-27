import sys
sys.path.append('.')

from datasets import load_dataset
import random
from tqdm import tqdm

from masked_bert import BertNER
from eval import Eval

def main(dataset="conll", split="test"):
    # Create NER comparison object
    ner_compare = BertNER()

    Eval.evaluate_dataset(ner_compare, dataset, split)

if __name__ == "__main__":
    main(dataset="conll", split="test")
    main(dataset="conll", split="train")
    main(dataset="wikiann", split="test")
    main(dataset="wikiann", split="train")