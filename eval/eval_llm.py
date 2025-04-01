import sys
sys.path.append('.')

import os
import yaml

from datasets import load_dataset
from tqdm import tqdm

from llm import llmNER 
from eval import Eval

def main(dataset="conll", split="test", few_shot=True):
    # Load configuration from config.yaml
    api_key = None
    base_url = None
    model_name = "gemma-3-27b-it"
    rate_limit = 0.3
    few_shot = few_shot

    # Create NER comparison object with rate limiting
    ner_compare = llmNER(model_name=model_name, api_key=api_key, base_url=base_url, rate=rate_limit, few_shot=few_shot)

    Eval.evaluate_dataset(ner_compare, dataset, split)
    print()
    print("Failed API calls: ", ner_compare.fail_count)

if __name__ == "__main__":
    main(dataset="conll", split="test", few_shot=False)
    main(dataset="conll", split="test", few_shot=True)
    # main(dataset="conll", split="train")
    # main(dataset="wikiann", split="test")
    # main(dataset="wikiann", split="train")