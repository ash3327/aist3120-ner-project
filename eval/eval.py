import sys
sys.path.append('.')

from datasets import load_dataset
import random
from tqdm import tqdm

from spacy_ner import SpacyNER
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
    def sample_f1_score(pred_entities, tokens, dataset_labels):
        """
        Calculate precision, recall, and F1 score for named entity recognition.
        
        Args:
            pred_entities (list): List of [label, text (entity)]
            tokens (list): List of tokens/words
            
        Returns:
            tuple: (precision, recall, f1_score, true_positives, false_positives, false_negatives)
        """
        """
        Compare NER labels between dataset and spaCy.
        
        Args:
            tokens (list): List of tokens/words
            dataset_labels (list): List of dataset label indices
            
        Returns:
            list: List of [token_idx, token, dataset_label_idx, spacy_label_idx]
        """
        # Combine everything into a list of [idx, token, dataset_label_idx, spacy_label_idx]
        result = []
        for i, (token, dataset_label_idx) in enumerate(zip(tokens, dataset_labels)):
            result.append([i, token, dataset_label_idx])

        # Extract entities from gold standard and predictions
        gold_entities = []
        i = 0
        while i < len(result):
            token_idx, token, gold_label_idx = result[i]
            
            # Process gold entities
            if gold_label_idx != 0: # not 'O' class
                gold_label = Eval.idx_to_label[gold_label_idx]
                if gold_label.startswith('B-'):
                    entity_type = gold_label[2:]
                    start_idx = token_idx
                    entity_tokens = [token]
                    
                    j = i + 1
                    while j < len(result):
                        next_token_idx, next_token, next_gold_label_idx = result[j]
                        if next_gold_label_idx != 0:
                            next_gold_label = Eval.idx_to_label[next_gold_label_idx]
                            if next_gold_label == f"I-{entity_type}":
                                entity_tokens.append(next_token)
                                j += 1
                                continue
                        break
                    
                    end_idx = start_idx + len(entity_tokens)
                    gold_entities.append((entity_type, ' '.join(tokens[start_idx:end_idx])))
            
            i += 1

        # print("Gold entities:", gold_entities)
        # print("Predicted entities:", pred_entities)
        
        # Calculate true positives (entities that match exactly in type and span)
        true_positives = sum(1 for entity in gold_entities if entity in pred_entities)
        false_positives = len(pred_entities) - true_positives
        false_negatives = len(gold_entities) - true_positives

        # fp, fn = [entity for entity in pred_entities if entity not in gold_entities], [entity for entity in gold_entities if entity not in pred_entities]
        # if fp or fn:
        #     print()
        #     print(tokens)
        #     print("False Positives", fp)
        #     print("False Negatives", fn)
        #     print("Predicted entities:", pred_entities)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        # Return TP, FP, FN along with metrics
        return precision, recall, f1_score, true_positives, false_positives, false_negatives
    
    @staticmethod
    def overall_f1_score(total_tp, total_fp, total_fn):
        """
        Calculate overall precision, recall, and F1 score based on cumulative counts.
        
        Args:
            total_tp (int): Total true positives across all samples
            total_fp (int): Total false positives across all samples
            total_fn (int): Total false negatives across all samples
            
        Returns:
            tuple: (overall_precision, overall_recall, overall_f1_score)
        """
        overall_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if overall_precision + overall_recall > 0 else 0
        
        return overall_precision, overall_recall, overall_f1
    
    @staticmethod
    def evaluate(ner_compare:NER, test_samples):
        """
        ner_compare should provide the method get_entities.
        """        
        sample_size = len(test_samples)
        # Initialize counters for overall metrics
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for idx, example in enumerate(tqdm(test_samples)):
            tokens = example["tokens"]
            ner_tags = example["ner_tags"]
            
            # print("\n" + "=" * 50)
            # print(f"\nExample {idx+1}:")
            # print(f"Text: {' '.join(tokens)}")
            
            # Get entity comparisons
            pred_entities = ner_compare.get_entities(tokens)
            
            # Print comparison table with 4 columns as requested
            # print("\nLocation    Word        Label       Prediction")
            # print("-" * 50)
            # for loc, word, label_idx, pred_idx in entity_comparisons:
            #     # Convert indices to text labels only for printing
            #     label_text = Eval.idx_to_label.get(label_idx, "Unknown")
            #     pred_text = Eval.idx_to_label.get(pred_idx, "Unknown")
            #     # print(f"{loc:<12}{word:<12}{label_text:<12}{pred_text}")

            # Calculate F1 score for this sample
            precision, recall, f1_score, tp, fp, fn = Eval.sample_f1_score(pred_entities, tokens, ner_tags)
            # Accumulate counters
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            # print("-" * 50)
            # print(f"Sample Precision: {precision:.4f}")
            # print(f"Sample Recall: {recall:.4f}")
            # print(f"Sample F1 Score: {f1_score:.4f}")
        
        # Calculate overall metrics using the new method
        overall_precision, overall_recall, overall_f1 = Eval.overall_f1_score(
            total_tp, total_fp, total_fn
        )
        
        print("\n" + "=" * 50)
        print(f"\nOVERALL EVALUATION (across {sample_size} samples):")
        print(f"True Positives: {total_tp}")
        print(f"False Positives: {total_fp}")
        print(f"False Negatives: {total_fn}")
        print(f"Overall Precision: {overall_precision:.4f}")
        print(f"Overall Recall: {overall_recall:.4f}")
        print(f"Overall F1 Score: {overall_f1:.4f}")

    @staticmethod
    def evaluate_dataset(ner_compare:NER, dataset="conll", split="test"):
        print("\n"+("="*50))
        print(f"Evaluating {dataset} dataset ({split})...")
        print("="*50)
        if dataset == "conll":
            # Load the conll2003 dataset
            print(f"Loading conll2003 dataset ({split})...")
            dataset = load_dataset("eriktks/conll2003")

            # Process examples from the test set
            test_samples = dataset[split] #.select(random.sample(range(len(dataset["test"])), sample_size))
        elif dataset == "wikiann":
            # Load the wikiann dataset
            print(f"Loading wikiann dataset ({split})...")
            dataset = load_dataset("unimelb-nlp/wikiann", "en", trust_remote_code=True)

            # Process examples from the test set
            test_samples = dataset[split] #.select(random.sample(range(len(dataset["test"])), sample_size))
        
        Eval.evaluate(ner_compare, test_samples)

def main(dataset="conll", split="test"):
    # Create NER comparison object
    ignore_types = ['DATE', 'TIME', 'CARDINAL', 'ORDINAL', 'MONEY', 'PERCENT', 'QUANTITY', 'PRODUCT', 'WORK_OF_ART']
    ner_compare = SpacyNER(ignore_types=ignore_types)

    Eval.evaluate_dataset(ner_compare, dataset, split)

if __name__ == "__main__":
    main(split="test")
