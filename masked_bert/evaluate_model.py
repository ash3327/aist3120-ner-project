"""
python masked_bert/evaluate_model.py --checkpoint_path runs/bert_ft_v4
"""

import sys
sys.path.append('.')

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from train_masked_bert import MaskedNERDataset, compute_metrics

def evaluate_model(model, tokenizer, dataset='conll', split='test'):
    # Load dataset based on the dataset_name
    if dataset == "conll":
        print(f"Loading conll2003 dataset ({split})...")
        raw_dataset = load_dataset("eriktks/conll2003")
        test_samples = raw_dataset[split]
    elif dataset == "wikiann":
        print(f"Loading wikiann dataset ({split})...")
        raw_dataset = load_dataset("unimelb-nlp/wikiann", "en", trust_remote_code=True)
        test_samples = raw_dataset[split]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    # Create test dataset
    test_dataset = MaskedNERDataset(raw_dataset['test'], tokenizer, mask_probability=0.0)  # No masking for test set
    
    # Training arguments (only for evaluation)
    training_args = TrainingArguments(
        output_dir="temp_eval_output",  # Temporary directory for evaluation
        per_device_eval_batch_size=16,
        logging_dir="temp_eval_logs",  # Temporary log directory
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    # Evaluate
    print(f"Evaluating the model on {dataset} dataset...")
    final_metrics = trainer.evaluate()
    print(f"\nFinal test metrics for {dataset}:", final_metrics)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    
    # Load model and tokenizer once
    print(f"Loading model and tokenizer from checkpoint: {args.checkpoint_path}")
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    
    # Evaluate on each dataset
    evaluate_model(model=model, tokenizer=tokenizer, dataset='conll', split='test')
    evaluate_model(model=model, tokenizer=tokenizer, dataset='wikiann', split='test')