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
    
    def get_entities(self, tokens):
        """
        Compare NER labels between dataset and spaCy.
        
        Args:
            tokens (list): List of tokens/words
            
        Returns:
            list: List of [label, text] tuples
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
        
        # Combine everything into a list of [idx, token, spacy_label_idx]
        result = []
        for i, (token, spacy_label_idx) in enumerate(zip(tokens, spacy_label_indices)):
            result.append([i, token, spacy_label_idx])
        
        pred_entities = []
        
        # Process gold entities
        i = 0
        while i < len(result):
            token_idx, token, pred_label_idx = result[i]
            
            # Process predicted entities
            if pred_label_idx != 0:
                pred_label = self.idx_to_label[pred_label_idx]
                if pred_label.startswith('B-'):
                    entity_type = pred_label[2:]
                    start_idx = token_idx
                    entity_tokens = [token]
                    
                    j = i + 1
                    while j < len(result):
                        next_token_idx, next_token, next_pred_label_idx = result[j]
                        if next_pred_label_idx != 0:
                            next_pred_label = self.idx_to_label[next_pred_label_idx]
                            if next_pred_label == f"I-{entity_type}":
                                entity_tokens.append(next_token)
                                j += 1
                                continue
                        break
                    
                    end_idx = start_idx + len(entity_tokens)
                    pred_entities.append((entity_type, ' '.join(tokens[start_idx:end_idx])))
            
            i += 1

        return pred_entities