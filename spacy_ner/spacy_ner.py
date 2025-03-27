import sys
sys.path.append('.')

import spacy
from datasets import load_dataset
from spacy.tokens import Doc
import random

from libs import NER

class SpacyNER(NER):
    def __init__(self, model_name="en_core_web_md", ignore_types=None):
        self.nlp = spacy.load(model_name)
        self.ignore_types = ignore_types or []  # List of spaCy entity types to ignore
        # Map of CoNLL label indices to label names
        self.idx_to_label = {
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
        self.label_to_idx = {v: k for k, v in self.idx_to_label.items()}
        
        # Map spaCy entity types to CoNLL types
        self.spacy_to_conll = {
            "PERSON": "PER",
            "ORG": "ORG",
            "GPE": "LOC",
            "LOC": "LOC",
            "FAC": "LOC",
            "NORP": "MISC",
            "PRODUCT": "MISC",
            "EVENT": "MISC",
            "WORK_OF_ART": "MISC",
            "LAW": "MISC",
            "LANGUAGE": "MISC",
            "DATE": "MISC",
            "TIME": "MISC",
            "PERCENT": "MISC",
            "MONEY": "MISC",
            "QUANTITY": "MISC",
            "CARDINAL": "MISC",
            "ORDINAL": "MISC"
        }
    
    def get_entities(self, tokens, dataset_labels):
        """
        Compare NER labels between dataset and spaCy.
        
        Args:
            tokens (list): List of tokens/words
            dataset_labels (list): List of dataset label indices
            
        Returns:
            list: List of [token_idx, token, dataset_label_idx, spacy_label_idx]
        """
        # Create a custom Doc with the tokens from the dataset
        words = tokens
        spaces = [True] * (len(words) - 1) + [False]
        doc = Doc(self.nlp.vocab, words=words, spaces=spaces)
        
        # Process with spaCy pipeline (skip the tokenizer)
        for name, proc in self.nlp.pipeline:
            doc = proc(doc)
        
        # Initialize spaCy predictions to 0 (O - outside)
        spacy_label_indices = [0] * len(tokens)
        
        # Extract entity labels from spaCy's predictions
        for ent in doc.ents:
            # Skip entity types that should be ignored
            if ent.label_ in self.ignore_types:
                continue
                
            # Convert spaCy entity type to CoNLL format
            entity_type = self.spacy_to_conll.get(ent.label_, "MISC")
            
            # Mark entity tokens with B- or I- prefix and convert to index
            for i, token_idx in enumerate(range(ent.start, ent.end)):
                if token_idx < len(tokens):  # Safety check
                    prefix = "B-" if i == 0 else "I-"
                    label_text = f"{prefix}{entity_type}"
                    spacy_label_indices[token_idx] = self.label_to_idx.get(label_text, 0)
        
        # Combine everything into a list of [idx, token, dataset_label_idx, spacy_label_idx]
        result = []
        for i, (token, dataset_label_idx, spacy_label_idx) in enumerate(zip(tokens, dataset_labels, spacy_label_indices)):
            result.append([i, token, dataset_label_idx, spacy_label_idx])
        
        return result

    def sample_f1_score(self, entity_comparisons):
        """
        Calculate precision, recall, and F1 score for named entity recognition.
        
        Args:
            entity_comparisons (list): List of [token_idx, token, dataset_label_idx, spacy_label_idx]
            
        Returns:
            tuple: (precision, recall, f1_score, true_positives, false_positives, false_negatives)
        """
        # Extract entities from gold standard and predictions
        gold_entities = []
        pred_entities = []
        i = 0
        while i < len(entity_comparisons):
            token_idx, token, gold_label_idx, pred_label_idx = entity_comparisons[i]
            
            # Process gold entities
            if gold_label_idx != 0:
                gold_label = self.idx_to_label[gold_label_idx]
                if gold_label.startswith('B-'):
                    entity_type = gold_label[2:]
                    start_idx = token_idx
                    entity_tokens = [token]
                    
                    j = i + 1
                    while j < len(entity_comparisons):
                        next_token_idx, next_token, next_gold_label_idx, _ = entity_comparisons[j]
                        if next_gold_label_idx != 0:
                            next_gold_label = self.idx_to_label[next_gold_label_idx]
                            if next_gold_label == f"I-{entity_type}":
                                entity_tokens.append(next_token)
                                j += 1
                                continue
                        break
                    
                    end_idx = start_idx + len(entity_tokens)
                    gold_entities.append((entity_type, start_idx, end_idx))
            
            # Process predicted entities
            if pred_label_idx != 0:
                pred_label = self.idx_to_label[pred_label_idx]
                if pred_label.startswith('B-'):
                    entity_type = pred_label[2:]
                    start_idx = token_idx
                    entity_tokens = [token]
                    
                    j = i + 1
                    while j < len(entity_comparisons):
                        next_token_idx, next_token, _, next_pred_label_idx = entity_comparisons[j]
                        if next_pred_label_idx != 0:
                            next_pred_label = self.idx_to_label[next_pred_label_idx]
                            if next_pred_label == f"I-{entity_type}":
                                entity_tokens.append(next_token)
                                j += 1
                                continue
                        break
                    
                    end_idx = start_idx + len(entity_tokens)
                    pred_entities.append((entity_type, start_idx, end_idx))
            
            i += 1

        # print("Gold entities:", gold_entities)
        # print("Predicted entities:", pred_entities)
        
        # Calculate true positives (entities that match exactly in type and span)
        true_positives = sum(1 for entity in gold_entities if entity in pred_entities)
        false_positives = len(pred_entities) - true_positives
        false_negatives = len(gold_entities) - true_positives
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        # Return TP, FP, FN along with metrics
        return precision, recall, f1_score, true_positives, false_positives, false_negatives
    
    def overall_f1_score(self, total_tp, total_fp, total_fn):
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

def main():
    # Load the conll2003 dataset
    print("Loading conll2003 dataset...")
    dataset = load_dataset("eriktks/conll2003")
    
    # Create NER comparison object
    ignore_types = ['DATE', 'TIME', 'CARDINAL', 'ORDINAL', 'MONEY', 'PERCENT', 'QUANTITY', 'PRODUCT', 'WORK_OF_ART']
    ner_compare = SpacyNER(ignore_types=ignore_types)
    
    # Process examples from the test set
    sample_size = 1000
    test_samples = dataset["test"].select(random.sample(range(len(dataset["test"])), sample_size))
    
    # Initialize counters for overall metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for idx, example in enumerate(test_samples):
        tokens = example["tokens"]
        ner_tags = example["ner_tags"]
        
        print("\n" + "=" * 50)
        print(f"\nExample {idx+1}:")
        print(f"Text: {' '.join(tokens)}")
        
        # Get entity comparisons
        entity_comparisons = ner_compare.get_entities(tokens, ner_tags)
        
        # Print comparison table with 4 columns as requested
        print("\nLocation    Word        Label       Prediction")
        print("-" * 50)
        for loc, word, label_idx, pred_idx in entity_comparisons:
            # Convert indices to text labels only for printing
            label_text = ner_compare.idx_to_label.get(label_idx, "Unknown")
            pred_text = ner_compare.idx_to_label.get(pred_idx, "Unknown")
            print(f"{loc:<12}{word:<12}{label_text:<12}{pred_text}")

        # Calculate F1 score for this sample
        precision, recall, f1_score, tp, fp, fn = ner_compare.sample_f1_score(entity_comparisons)
        # Accumulate counters
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        print("-" * 50)
        print(f"Sample Precision: {precision:.4f}")
        print(f"Sample Recall: {recall:.4f}")
        print(f"Sample F1 Score: {f1_score:.4f}")
    
    # Calculate overall metrics using the new method
    overall_precision, overall_recall, overall_f1 = ner_compare.overall_f1_score(
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
    
if __name__ == "__main__":
    main()

