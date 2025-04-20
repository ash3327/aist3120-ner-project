import sys
sys.path.append('.')

from datasets import load_dataset
import random
from tqdm import tqdm
import argparse

from spacy_ner import SpacyNER
from eval import Eval

def main(dataset="conll", split="test", output_file="records/spacy.json", model_name="en_core_web_md"):
    # Create NER comparison object
    ignore_types = ['DATE', 'TIME', 'CARDINAL', 'ORDINAL', 'MONEY', 'PERCENT', 'QUANTITY', 'PRODUCT', 'WORK_OF_ART']
    ner_compare = SpacyNER(ignore_types=ignore_types, model_name=model_name)

    Eval.evaluate_dataset(ner_compare, dataset, split, output_file=output_file)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate spaCy NER model on datasets')
    parser.add_argument('--dataset', type=str, default="conll", choices=["conll", "wikiann"], 
                        help='Dataset to evaluate on (default: conll)')
    parser.add_argument('--split', type=str, default="test", choices=["test", "validation", "train"],
                        help='Dataset split to use (default: test)')
    parser.add_argument('--output_file', type=str, default="records/spacy.json",
                        help='Path to output file (default: records/spacy.json)')
    parser.add_argument('--model_name', type=str, default="en_core_web_md",
                        help='SpaCy model to use (default: en_core_web_md)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no arguments were provided, run the default evaluations
    if len(sys.argv) == 1:
        model_name = "en_core_web_md"  # Default model
        main(dataset="conll", split="test", model_name=model_name)
        # main(dataset="conll", split="validation", model_name=model_name)
        # main(dataset="conll", split="train", model_name=model_name)
        main(dataset="wikiann", split="test", model_name=model_name)
        # main(dataset="wikiann", split="validation", model_name=model_name)
        # main(dataset="wikiann", split="train", model_name=model_name)
    else:
        # Run with provided arguments
        main(
            dataset=args.dataset,
            split=args.split,
            output_file=args.output_file,
            model_name=args.model_name
        )