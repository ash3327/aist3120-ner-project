from datasets import load_dataset
from tqdm import tqdm

def get_dataset(dataset="conll", split="test"):
    print("\n"+("="*50))
    print(f"Getting {dataset} dataset ({split})...")
    print("="*50)
    if dataset == "conll":
        # Load the conll2003 dataset
        print(f"Loading conll2003 dataset ({split})...")
        dataset = load_dataset("eriktks/conll2003")
        print(dataset['test'][123])

        # Process examples from the test set
        test_samples = dataset[split] #.select(random.sample(range(len(dataset["test"])), sample_size))
    elif dataset == "wikiann":
        # Load the wikiann dataset
        print(f"Loading wikiann dataset ({split})...")
        dataset = load_dataset("unimelb-nlp/wikiann", "en", trust_remote_code=True)

        # Process examples from the test set
        test_samples = dataset[split] #.select(random.sample(range(len(dataset["test"])), sample_size))
    
    for idx, example in enumerate(tqdm(test_samples)):
        tokens = example["tokens"]
        ner_tags = example["ner_tags"] # same length
        print(tokens, ner_tags)
        break
        

if __name__ == "__main__":
    get_dataset(dataset="conll", split="test")
    # get_dataset(dataset="conll", split="train")
    # get_dataset(dataset="wikiann", split="test")
    # get_dataset(dataset="wikiann", split="train")