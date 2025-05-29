import sys
sys.path.append('.')

from datasets import load_dataset
import random
from tqdm import tqdm
import os
import json
import datetime
from collections import defaultdict

from libs import NER

class Eval:
    # Map of CoNLL label indices to label names
    idx_to_label = {
        0: "O",      # Outside of a named entity
        1: "B-PER",  # Beginning of a person name
        2: "I-PER",  # Inside of a person name
        3: "B-ORG",  # Beginning of an organization name
        4: "I-ORG",  # Inside of an organization name
        5: "B-LOC",  # Beginning of a location name
        6: "I-LOC",  # Inside of a location name
        7: "B-MISC", # Beginning of a miscellaneous entity
        8: "I-MISC"  # Inside of a miscellaneous entity
    }
    
    # Create reverse mapping
    label_to_idx = {v: k for k, v in idx_to_label.items()}
    
    @staticmethod
    def get_seen_entities(dataset, split="train", reference_set=None):
        """
        Extract entities from a dataset split and optionally check against a reference set.
        
        Args:
            dataset: The dataset object
            split: The split to use (default: "train")
            reference_set: Optional set of (entity_type, entity_text) tuples to check against
            
        Returns:
            If reference_set is None:
                set: Set of (entity_type, entity_text) tuples that appear in the split
            If reference_set is provided:
                tuple: (total_entities, unseen_entities) where:
                    total_entities: int, total number of entity occurrences
                    unseen_entities: int, number of entities not in reference_set
        """
        entities = set()
        total_count = 0
        unseen_count = 0
        
        for example in dataset[split]:
            tokens = example["tokens"]
            ner_tags = example["ner_tags"]
            
            # Extract entities using the same logic as in sample_f1_score
            i = 0
            while i < len(tokens):
                if ner_tags[i] != 0:  # not 'O' class
                    gold_label = Eval.idx_to_label[ner_tags[i]]
                    if gold_label.startswith('B-'):
                        entity_type = gold_label[2:]
                        start_idx = i
                        entity_tokens = [tokens[i]]
                        
                        j = i + 1
                        while j < len(tokens):
                            if ner_tags[j] != 0:
                                next_gold_label = Eval.idx_to_label[ner_tags[j]]
                                if next_gold_label == f"I-{entity_type}":
                                    entity_tokens.append(tokens[j])
                                    j += 1
                                    continue
                            break
                        
                        entity_text = ' '.join(entity_tokens)
                        entity = (entity_type, entity_text)
                        
                        if reference_set is None:
                            entities.add(entity)
                        else:
                            total_count += 1
                            if entity not in reference_set:
                                unseen_count += 1
                i += 1
        
        if reference_set is None:
            return entities
        else:
            return total_count, unseen_count

    @staticmethod
    def sample_f1_score(pred_entities, tokens, dataset_labels, seen_entities=None):
        """
        Calculate precision, recall, and F1 score for named entity recognition.
        If seen_entities is provided, also calculate separate metrics for seen and unseen entities.
        
        Args:
            pred_entities (list): List of [label, text (entity)]
            tokens (list): List of tokens/words
            dataset_labels (list): List of dataset label indices
            seen_entities (set): Set of (entity_type, entity_text) tuples that appear in training
            
        Returns:
            tuple: (metrics_dict) containing precision, recall, f1_score, and counts for all/seen/unseen entities
        """
        # Extract entities from gold standard and predictions
        gold_entities = []
        i = 0
        while i < len(tokens):
            token_idx, token, gold_label_idx = i, tokens[i], dataset_labels[i]
            
            # Process gold entities
            if gold_label_idx != 0: # not 'O' class
                gold_label = Eval.idx_to_label[gold_label_idx]
                if gold_label.startswith('B-'):
                    entity_type = gold_label[2:]
                    start_idx = token_idx
                    entity_tokens = [token]
                    
                    j = i + 1
                    while j < len(tokens):
                        next_token_idx, next_token, next_gold_label_idx = j, tokens[j], dataset_labels[j]
                        if next_gold_label_idx != 0:
                            next_gold_label = Eval.idx_to_label[next_gold_label_idx]
                            if next_gold_label == f"I-{entity_type}":
                                entity_tokens.append(next_token)
                                j += 1
                                continue
                        break
                    
                    end_idx = start_idx + len(entity_tokens)
                    entity_text = ' '.join(tokens[start_idx:end_idx])
                    gold_entities.append((entity_type, entity_text))
            
            i += 1

        # Calculate metrics for all entities
        true_positives = sum(1 for entity in gold_entities if entity in pred_entities)
        false_positives = len(pred_entities) - true_positives
        false_negatives = len(gold_entities) - true_positives

        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        metrics = {
            "all": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "tp": true_positives,
                "fp": false_positives,
                "fn": false_negatives
            }
        }

        # If seen_entities is provided, calculate separate metrics for seen and unseen entities
        if seen_entities is not None:
            # Split gold entities into seen and unseen
            seen_gold = [e for e in gold_entities if e in seen_entities]
            unseen_gold = [e for e in gold_entities if e not in seen_entities]
            
            # Split predicted entities into seen and unseen
            seen_pred = [e for e in pred_entities if e in seen_entities]
            unseen_pred = [e for e in pred_entities if e not in seen_entities]
            
            # Calculate metrics for seen entities
            seen_tp = sum(1 for entity in seen_gold if entity in seen_pred)
            seen_fp = len(seen_pred) - seen_tp
            seen_fn = len(seen_gold) - seen_tp
            
            seen_precision = seen_tp / (seen_tp + seen_fp) if seen_tp + seen_fp > 0 else 0
            seen_recall = seen_tp / (seen_tp + seen_fn) if seen_tp + seen_fn > 0 else 0
            seen_f1 = 2 * seen_precision * seen_recall / (seen_precision + seen_recall) if seen_precision + seen_recall > 0 else 0
            
            # Calculate metrics for unseen entities
            unseen_tp = sum(1 for entity in unseen_gold if entity in unseen_pred)
            unseen_fp = len(unseen_pred) - unseen_tp
            unseen_fn = len(unseen_gold) - unseen_tp
            
            unseen_precision = unseen_tp / (unseen_tp + unseen_fp) if unseen_tp + unseen_fp > 0 else 0
            unseen_recall = unseen_tp / (unseen_tp + unseen_fn) if unseen_tp + unseen_fn > 0 else 0
            unseen_f1 = 2 * unseen_precision * unseen_recall / (unseen_precision + unseen_recall) if unseen_precision + unseen_recall > 0 else 0
            
            metrics.update({
                "seen": {
                    "precision": seen_precision,
                    "recall": seen_recall,
                    "f1_score": seen_f1,
                    "tp": seen_tp,
                    "fp": seen_fp,
                    "fn": seen_fn
                },
                "unseen": {
                    "precision": unseen_precision,
                    "recall": unseen_recall,
                    "f1_score": unseen_f1,
                    "tp": unseen_tp,
                    "fp": unseen_fp,
                    "fn": unseen_fn
                }
            })
        
        return metrics

    @staticmethod
    def evaluate(ner_compare:NER, test_samples, output_file=None, seen_entities=None):
        """
        ner_compare should provide the method get_entities.
        """        
        sample_size = len(test_samples)
        
        # Initialize counters for overall metrics
        total_metrics = {
            "all": {"tp": 0, "fp": 0, "fn": 0},
            "seen": {"tp": 0, "fp": 0, "fn": 0},
            "unseen": {"tp": 0, "fp": 0, "fn": 0}
        }
        
        # Create records structure for logging results
        records = []
        
        for idx, example in enumerate(tqdm(test_samples)):
            tokens = example["tokens"]
            ner_tags = example["ner_tags"]
            
            # Get entity predictions
            pred_entities = ner_compare.get_entities(tokens)
            
            # Calculate metrics for this sample
            sample_metrics = Eval.sample_f1_score(pred_entities, tokens, ner_tags, seen_entities)
            
            # Accumulate counters
            for category in total_metrics:
                for metric in ["tp", "fp", "fn"]:
                    total_metrics[category][metric] += sample_metrics[category][metric]
            
            # Save data for this sample to records
            record = {
                "tokens": tokens,
                "ner_tags": ner_tags,
                "predicted_entities": pred_entities,
                "metrics": sample_metrics
            }
            records.append(record)
        
        # Calculate overall metrics
        overall_metrics = {}
        for category in total_metrics:
            tp = total_metrics[category]["tp"]
            fp = total_metrics[category]["fp"]
            fn = total_metrics[category]["fn"]
            
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            
            overall_metrics[category] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn
            }
        
        # Print results
        print("\n" + "=" * 50)
        print(f"\nOVERALL EVALUATION (across {sample_size} samples):")
        
        for category in ["all", "seen", "unseen"]:
            metrics = overall_metrics[category]
            print(f"\n{category.upper()} ENTITIES:")
            print(f"True Positives: {metrics['tp']}")
            print(f"False Positives: {metrics['fp']}")
            print(f"False Negatives: {metrics['fn']}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        # Save records to file if output_file is provided
        if output_file:
            # Ensure records directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Create a record summary with metadata and results
            record_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "model": ner_compare.__class__.__name__, 
                "sample_size": sample_size,
                "overall_metrics": overall_metrics,
                "samples": records
            }
            
            # Write to JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(record_data, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def evaluate_dataset(ner_compare:NER, dataset="conll", split="test", output_file=None):
        print("\n"+("="*50))
        print(f"Evaluating {dataset} dataset ({split})...")
        print("="*50)
        
        # Load the dataset
        train_dataset = load_dataset("eriktks/conll2003")
        if dataset == "conll":
            print(f"Loading conll2003 dataset...")
            dataset = load_dataset("eriktks/conll2003")
        elif dataset == "wikiann":
            print(f"Loading wikiann dataset...")
            dataset = load_dataset("unimelb-nlp/wikiann", "en", trust_remote_code=True)
        
        # Get seen entities from training set
        seen_entities = Eval.get_seen_entities(train_dataset)
        print(f"Found {len(seen_entities)} unique entities in training set")
        
        # Count entities in test set and check against training set
        total_test_entities, unseen_test_entities = Eval.get_seen_entities(dataset, split, seen_entities)
        print(f"Found {total_test_entities} total entities in test set")
        print(f"Of which {unseen_test_entities} are unseen (not in training set)")
        print(f"Unseen entity ratio: {unseen_test_entities/total_test_entities:.2%}")
        
        # Process examples from the test set
        test_samples = dataset[split]
        
        Eval.evaluate(ner_compare, test_samples, output_file, seen_entities)
    
    @staticmethod
    def evaluate_record(record_file):
        """
        Evaluate NER performance by reading saved records from a file.
        
        Args:
            record_file (str): Path to the record file
            
        Returns:
            dict: Dictionary containing overall metrics for all/seen/unseen entities
        """
        print("\n"+("="*50))
        print(f"Evaluating from record file: {record_file}")
        print("="*50)
        
        # Read the saved record file
        try:
            with open(record_file, 'r', encoding='utf-8') as f:
                record_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading record file: {e}")
            return None
        
        overall_metrics = record_data["overall_metrics"]
        
        for category in ["all", "seen", "unseen"]:
            metrics = overall_metrics[category]
            print(f"\n{category.upper()} ENTITIES:")
            print(f"True Positives: {metrics['tp']}")
            print(f"False Positives: {metrics['fp']}")
            print(f"False Negatives: {metrics['fn']}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        return overall_metrics

def main(dataset="conll", split="test"):
    # Create NER comparison object
    from spacy_ner import SpacyNER
    ignore_types = ['DATE', 'TIME', 'CARDINAL', 'ORDINAL', 'MONEY', 'PERCENT', 'QUANTITY', 'PRODUCT', 'WORK_OF_ART']
    ner_compare = SpacyNER(ignore_types=ignore_types)

    Eval.evaluate_dataset(ner_compare, dataset, split)

if __name__ == "__main__":
    main(split="test")
