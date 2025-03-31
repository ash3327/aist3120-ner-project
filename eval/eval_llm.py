import sys
sys.path.append('.')

import os
import yaml

from datasets import load_dataset
from tqdm import tqdm

from llm import llmNER
from eval import Eval

def main(dataset="conll", split="test"):
    # Load configuration from config.yaml
    api_key = None
    base_url = None
    model_name = "gemma-3-27b-it"
    rate_limit = 0.4 
    few_shot = True

    # Create NER comparison object with rate limiting
    ner_compare = llmNER(model_name=model_name, api_key=api_key, base_url=base_url, rate=rate_limit, few_shot=few_shot)

    Eval.evaluate_dataset(ner_compare, dataset, split)

if __name__ == "__main__":
    main(dataset="conll", split="test")
    # main(dataset="conll", split="train")
    # main(dataset="wikiann", split="test")
    # main(dataset="wikiann", split="train")