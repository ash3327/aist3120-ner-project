import sys
import os
import random
from datasets import load_dataset
from itertools import combinations

# Add the parent directory to the path to import SpacyNER
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spacy_ner.spacy_ner import SpacyNER

def test_all_misc_combinations():
    """
    Test different mapping strategies by trying all possible combinations of 
    what MISC entity types to include or ignore.
    """
    print("Loading conll2003 dataset...")
    dataset = load_dataset("eriktks/conll2003")
    
    # Set a seed for reproducibility
    # random.seed(42)
    
    # Create a consistent test sample
    sample_size = 2000
    test_samples = dataset["test"].select(random.sample(range(len(dataset["test"])), sample_size))
    
    # Get a default SpacyNER to extract misc_types
    default_ner = SpacyNER()
    
    # Extract all entity types that map to MISC in the default mapping
    misc_types = [ent_type for ent_type, conll_type in default_ner.spacy_to_conll.items() 
                 if conll_type == "MISC"]
    
    # Store results for all combinations
    results = {}
    
    # Start with default mapping (no types ignored)
    default_strategy = "default_all_misc"
    print(f"\n\n{'=' * 60}")
    print(f"Testing strategy: {default_strategy}")
    print(f"Ignoring: []")
    print('=' * 60)
    
    ner_model = SpacyNER()
    
    # Initialize counters
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Process each sample
    for example in test_samples:
        tokens = example["tokens"]
        ner_tags = example["ner_tags"]
        
        # Get entity comparisons
        entity_comparisons = ner_model.get_entities(tokens, ner_tags)
        
        # Calculate F1 score for this sample
        _, _, _, tp, fp, fn = ner_model.sample_f1_score(entity_comparisons)
        
        # Accumulate counters
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Calculate overall metrics
    overall_precision, overall_recall, overall_f1 = ner_model.overall_f1_score(
        total_tp, total_fp, total_fn
    )
    
    print(f"True Positives: {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"False Negatives: {total_fn}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")
    
    # Store results
    results[default_strategy] = {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "ignore_types": []
    }
    
    # Define logical groupings of entity types to make testing more manageable
    type_groups = {
        "temporal": ["DATE", "TIME"],
        "numeric": ["CARDINAL", "ORDINAL"],
        "financial": ["MONEY", "PERCENT", "QUANTITY"],
        "cultural": ["NORP", "LANGUAGE"],
        "creative": ["PRODUCT", "WORK_OF_ART"],
        "institutional": ["EVENT", "LAW"]
    }
    
    # Test all combinations of these groups
    for r in range(1, len(type_groups) + 1):
        for group_combo in combinations(type_groups.keys(), r):
            # Create the list of types to ignore for this combination
            ignore_types = []
            for group in group_combo:
                ignore_types.extend(type_groups[group])
            
            strategy_name = f"ignore_{'_'.join(group_combo)}"
            print(f"\n\n{'=' * 60}")
            print(f"Testing strategy: {strategy_name}")
            print(f"Ignoring: {ignore_types}")
            print('=' * 60)
            
            # Initialize SpacyNER with the current strategy
            ner_model = SpacyNER(ignore_types=ignore_types)
            
            # Initialize counters
            total_tp = 0
            total_fp = 0
            total_fn = 0
            
            # Process each sample
            for example in test_samples:
                tokens = example["tokens"]
                ner_tags = example["ner_tags"]
                
                # Get entity comparisons
                entity_comparisons = ner_model.get_entities(tokens, ner_tags)
                
                # Calculate F1 score for this sample
                _, _, _, tp, fp, fn = ner_model.sample_f1_score(entity_comparisons)
                
                # Accumulate counters
                total_tp += tp
                total_fp += fp
                total_fn += fn
            
            # Calculate overall metrics
            overall_precision, overall_recall, overall_f1 = ner_model.overall_f1_score(
                total_tp, total_fp, total_fn
            )
            
            print(f"True Positives: {total_tp}")
            print(f"False Positives: {total_fp}")
            print(f"False Negatives: {total_fn}")
            print(f"Overall Precision: {overall_precision:.4f}")
            print(f"Overall Recall: {overall_recall:.4f}")
            print(f"Overall F1 Score: {overall_f1:.4f}")
            
            # Store results
            results[strategy_name] = {
                "precision": overall_precision,
                "recall": overall_recall,
                "f1": overall_f1,
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "ignore_types": ignore_types
            }
    
    # Also test ignoring all MISC entities
    strategy_name = "ignore_all_misc"
    print(f"\n\n{'=' * 60}")
    print(f"Testing strategy: {strategy_name}")
    print(f"Ignoring: {misc_types}")
    print('=' * 60)
    
    ner_model = SpacyNER(ignore_types=misc_types)
    
    # Initialize counters
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Process each sample
    for example in test_samples:
        tokens = example["tokens"]
        ner_tags = example["ner_tags"]
        
        # Get entity comparisons
        entity_comparisons = ner_model.get_entities(tokens, ner_tags)
        
        # Calculate F1 score for this sample
        _, _, _, tp, fp, fn = ner_model.sample_f1_score(entity_comparisons)
        
        # Accumulate counters
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Calculate overall metrics
    overall_precision, overall_recall, overall_f1 = ner_model.overall_f1_score(
        total_tp, total_fp, total_fn
    )
    
    print(f"True Positives: {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"False Negatives: {total_fn}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")
    
    # Store results
    results[strategy_name] = {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "ignore_types": misc_types
    }
    
    # Compare results
    print("\n\n" + "=" * 90)
    print("COMPARISON OF MAPPING STRATEGIES")
    print("=" * 90)
    print(f"{'Strategy':<25} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 90)
    
    # Sort strategies by F1 score (highest first)
    sorted_strategies = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)
    
    for strategy_name, metrics in sorted_strategies:
        print(f"{strategy_name[:22]:<25} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
    
    # Find the best strategy
    best_strategy = sorted_strategies[0][0]
    best_ignore_types = sorted_strategies[0][1]["ignore_types"]
    print("\nBest strategy based on F1 score:", best_strategy)
    print("Ignored types:", best_ignore_types if best_ignore_types else "None")
    print(f"F1 Score: {sorted_strategies[0][1]['f1']:.4f}")

if __name__ == "__main__":
    test_all_misc_combinations()
