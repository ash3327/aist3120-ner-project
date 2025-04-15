import sys
sys.path.append('.')

import os
import numpy as np
import random
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer, 
    TrainingArguments
)
import evaluate
from eval.eval import Eval
import shutil  # Add this import for disk space checking

class NERDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        tokens = item['tokens']
        ner_tags = item['ner_tags']
        
        masked_tokens = tokens
        
        # Tokenize
        tokenized = self.tokenizer(
            masked_tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Align labels with word pieces
        labels = []
        word_ids = tokenized.word_ids()
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != previous_word_idx:  # First token of word
                labels.append(ner_tags[word_idx])
            else:  # Subsequent tokens of word
                labels.append(-100)  # Special value to ignore in loss
            previous_word_idx = word_idx
            
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': torch.tensor(labels)
        }

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[Eval.idx_to_label[l] for l in label if l != -100] 
                  for label in labels]
    true_predictions = [[Eval.idx_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
                       for prediction, label in zip(predictions, labels)]

    seqeval = evaluate.load("seqeval")
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def ensure_output_dir(output_dir):
    """Ensure the output directory exists and has sufficient space."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    total, used, free = shutil.disk_usage(output_dir)
    if free < 1e9:  # Less than 1GB free space
        raise RuntimeError(f"Insufficient disk space in {output_dir}. Free up space and try again.")

def main(checkpoint_path=None):
    # Set random seed
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Load dataset
    print("Loading CoNLL-2003 dataset...")
    raw_dataset = load_dataset("eriktks/conll2003")
    
    # =====================================================
    # Parameters Section
    # =====================================================
    base_model = "distilbert-base-cased" # v1,v2,v3,nomask_v1
    # base_model = "dslim/bert-base-NER" # v4
    require_reinitlialize = True # v1,v2,v3
    require_reinitlialize = True # v4

    output_dir = "runs/bert_ft_nomask_v1"
    ratio = 1 # v1-v4
    ratio = 3 # v5 
    num_epochs = 3 # v1-v4
    num_epochs = 3*ratio # v5
    num_epochs = 15 # nomask_v1
    batch_size = 32 # v1,v2,v3,v5
    # batch_size = 16 # v4
    lr, weight_decay = 2e-5, 0.01 # v1-v4
    lr, weight_decay = 2e-5/ratio, 0.01/ratio # v5
    lr, weight_decay = 5e-5, 1e-5 # nomask_v1
    # =====================================================
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Load model from checkpoint or initialize a new one
    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
    elif require_reinitlialize:
        print("Initializing new model...")
        model = AutoModelForTokenClassification.from_pretrained(
            base_model,
            num_labels=len(Eval.idx_to_label),
            id2label=Eval.idx_to_label,
            label2id=Eval.label_to_idx
        )
    else:
        print("Initializing new model...")
        model = AutoModelForTokenClassification.from_pretrained(base_model)
    
    # Ensure output directory is valid
    ensure_output_dir(output_dir)
    
    # Create datasets
    train_dataset = NERDataset(raw_dataset['train'], tokenizer)
    eval_dataset = NERDataset(raw_dataset['validation'], tokenizer)
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Training arguments
    ## https://datascience.stackexchange.com/questions/64583/what-are-the-good-parameter-ranges-for-bert-hyperparameters-while-finetuning-it
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,  # Ensure old checkpoints are overwritten
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,  # Don't push to hub
        save_total_limit=2,  # Keep only the best 2 checkpoints
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Optionally evaluate on test set
    test_dataset = NERDataset(raw_dataset['test'], tokenizer)  # No masking for test set
    final_metrics = trainer.evaluate(test_dataset)
    print("\nFinal test metrics:", final_metrics)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()
    main(checkpoint_path=args.checkpoint_path)
