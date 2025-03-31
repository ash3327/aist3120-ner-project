import sys
import argparse
sys.path.append('.')

from datasets import load_dataset
import random
from tqdm import tqdm

from masked_bert import MaskedBertNER
from eval import Eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Masked BERT NER Evaluation")
    parser.add_argument("--model_name", type=str, default="runs/bert_ft_v1", help="Path to the model")
    args = parser.parse_args()

    # Create NER comparison object
    ner_compare = MaskedBertNER(model_name=args.model_name)

    main = lambda **kwargs: Eval.evaluate_dataset(ner_compare, **kwargs)

    main(dataset="conll", split="test")
    main(dataset="conll", split="validation")
    main(dataset="conll", split="train")
    main(dataset="wikiann", split="test")
    main(dataset="wikiann", split="validation")
    main(dataset="wikiann", split="train")