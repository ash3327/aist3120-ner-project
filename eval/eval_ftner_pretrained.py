import sys
import argparse
sys.path.append('.')

from datasets import load_dataset
import random
from tqdm import tqdm

from masked_bert import BertNER
from eval import Eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuned BERT NER Evaluation")
    parser.add_argument("--model_name", type=str, default="dslim/bert-large-NER", help="Name or path of the model")
    parser.add_argument("--aggregation_strategy", type=str, default="max", help="Aggregation strategy for NER")
    args = parser.parse_args()

    # Create NER comparison object
    ner_compare = BertNER(model_name=args.model_name, aggregation_strategy=args.aggregation_strategy)

    main = lambda **kwargs: Eval.evaluate_dataset(ner_compare, **kwargs)

    main(dataset="conll", split="test")
    # main(dataset="conll", split="validation")
    # main(dataset="conll", split="train")
    main(dataset="wikiann", split="test")
    # main(dataset="wikiann", split="validation")
    # main(dataset="wikiann", split="train")