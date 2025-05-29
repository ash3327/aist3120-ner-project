# AIST3120: Named Entity Recognition

[![Final Report](https://img.shields.io/badge/Final%20Report-blue?style=for-the-badge)](AIST3120_Final_Project_Report.pdf) 
[![Project Description](https://img.shields.io/badge/Project%20Description-green?style=for-the-badge)](project_description.pdf) 

## Overview
This project aims to explore methodologies related to Named Entity Recognition.

### Methodology

The project evaluated three distinct deep learning architectures for Named Entity Recognition (NER):

* **SpaCy:** (Responsible: Square) Explored the key approaches used in the SpaCy library, including hash embeddings and trasition-based models.
* **BERT (Bidirectional Encoder Representations from Transformers):** (Responsible: Sam/Ash) Fine-tuned the `bert-base-cased` model over NER task, with following modifications:
    * **Aggregation Strategies:** Improved aggregation method to make sure that each input word is not split into parts in the output.
    * **Masking:** Masked by chance 15% of named entities in the training data and adjusted the leraning rate and number of epochs for the fine-tuning process to test if encouraging the utilization of nearby context could improve the generalizability over unseen named-entities.
* **Gemma 3 (Decoder LLM):** (Responsible: Square) Tested the performance by using decoder LLMs on NER tasks, guided using prompt engineering. Ensured JSON output. Evaluated on both zero-shot (only instructions) and few-shot (instructions plus examples) cases.

For more experimental data, please see the [report](/AIST3120_Final_Project_Report.pdf) for more information.

## Job Allocation

* Sam: BERT experiments
* Square: SpaCy and Gemma 3 experiments
* Quinson: Literature review and result analysis

## Installation

* **Prerequisites**
    ```
    # *Open a new virtual environment before execution.
    
    # Install packages:
    pip install -r requirements.txt
    ```
* **Dataset: CoNLL-2003** | [Source](https://huggingface.co/datasets/eriktks) | [Data Description](https://huggingface.co/datasets/eriktks/conll2003#dataset-structure) | [Examples (for each NER class)](https://www.clips.uantwerpen.be/conll2003/ner/lists/)
* **Dataset: WikiAnn** | [Source](https://huggingface.co/datasets/unimelb-nlp/wikiann) | [Data Description](https://huggingface.co/datasets/unimelb-nlp/wikiann#dataset-structure)
  * Notes: Both dataset share the common set of NER tags from 0 to 6, but class 7 (B-MISC) and 8 (I-MISC) is not available in this dataset.
  * They both share the same format in `tokens` and `ner_tags` keys within each entry.

## Reproducing Experiments

### Spacy
Download the required models:
```
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
```

Run experiments:
```
# Test conll dataset with en_core_web_md model
python eval/eval_spacy.py --dataset conll --split test --model_name en_core_web_md --output_file records/spacy_conll_md.json

# Test conll dataset with en_core_web_lg model
python eval/eval_spacy.py --dataset conll --split test --model_name en_core_web_lg --output_file records/spacy_conll_lg.json

# Test wikiann dataset with en_core_web_md model
python eval/eval_spacy.py --dataset wikiann --split test --model_name en_core_web_md --output_file records/spacy_wikiann_md.json

# Test wikiann dataset with en_core_web_lg model
python eval/eval_spacy.py --dataset wikiann --split test --model_name en_core_web_lg --output_file records/spacy_wikiann_lg.json
```

### Decoder LLM

Export environment variables OPENAI_API_BASE_URL and OPENAI_API_KEY:
```
export OPENAI_API_BASE_URL=<your_openai_api_base_url>
export OPENAI_API_KEY=<your_openai_api_key>
```

Run the experiment:
```
python llm/llm_ner.py --dataset conll --split test
python llm/llm_ner.py --dataset wikiann --split test #skipped in actual evals but should work
```

### BERT Models

Run the following commands for all runs on custom aggregation strategy (explained below):

```
python masked_bert/masked_bert_ner.py --model_name dslim/bert-base-NER

# Trained models
# For model files specifically for our runs please send an email to us
python masked_bert/masked_bert_ner.py --model_name runs/bert_ft_v1 
```

Run the following commands for default aggregation strategies (yes, the file is fine_tuned_ner.py because we tested on fine-tuning models first before changing it to base models):

```
# Run with various aggregation strategies
python masked_bert/fine_tuned_ner.py --model_name dslim/bert-base-NER --aggregation_strategy simple
python masked_bert/fine_tuned_ner.py --model_name dslim/bert-base-NER --aggregation_strategy first
python masked_bert/fine_tuned_ner.py --model_name dslim/bert-base-NER --aggregation_strategy average
python masked_bert/fine_tuned_ner.py --model_name dslim/bert-base-NER --aggregation_strategy max

# Test with different datasets
python masked_bert/fine_tuned_ner.py --model_name dslim/bert-base-NER --dataset wikiann --aggregation_strategy max
```

### Other References

* https://huggingface.co/docs/transformers/main/en/tasks/masked_language_modeling
* https://huggingface.co/docs/transformers/main/en/tasks/token_classification

## Metric

$\text{F-}\beta$ score, reduces to F-1 score when $\beta=1$:
$$F_\beta=\frac{(\beta^2+1)\times\text{precision}\times\text{recall}}{\beta^2\times\text{precision}+\text{recall}}$$

where, precision and recalls are evaluated on whether the named entity is being identified (and matches exactly as the one in the data file).

## Aggregation Methods

* Custom Decoding (Implemented by Square): Token-wise merging of class labels.
  * Problems: Sometimes words are truncated.
  * Example:
    ```
    False Positives [('PER', 'Shane War')]
    False Negatives [('PER', 'Shane Warne')]
    ```
* aggregation_strategy="max": Runs best among different aggregation strategies.
  * Problems: Since tokenization can be different, sometimes words that were in the same "token" in the input does not stay as the same token after Bert tokenization, which make aggregation fail to cover that.
  * Example: 
    ```python
    'Pau-Orthez' # Input token
    'Pau', 'Orthez' # Output ('-' is not classified as a part of the name)
    ```
* v3 aggregation strategy:
  * Custom aggregation strategy in `masked_bert_ner.py` that ensures that each input token is treated as a whole.
* Seqeval: Package for evaluation.

## Results (*raw)

Metric: Precision/Recall/F1 Score

**Aggregation Method Results**

| Model    | Precision | Recall  | F1      |
| -------- | --------- | ------- | ------- |
| Simple   | 0.5804    | 0.7350  | 0.6486  |
| First    | 0.8318    | 0.8890  | 0.8595  |
| Average  | 0.8368    | 0.8791  | 0.8574  |
| Max      | 0.8359    | 0.8817  | 0.8582  |
| Custom   | **0.9124**    | **0.9189**  | **0.9157**  |

**Masking Method Results**

| Model                     | Precision | Recall  | F1      |
| ------------------------- | --------- | ------- | ------- |
| **CoNLL-2003 (Test)**     |           |         |         |
| Distillbert-Cased fine-tune (no mask) | 0.8927    | **0.9032**  | **0.8979**  |
| Bert-Base-Cased fine-tune (masked) | **0.8929**    | 0.8948  | 0.8939  |
| **WikiAnn (Test) (Not trained on)** |           |         |         |
| Distillbert-Cased fine-tune (no mask) | **0.4603**    | **0.5134**  | **0.4854**  |
| Bert-Base-Cased (masked)  | 0.4577    | 0.5080  | 0.4816  |

**CoNLL-2003 Results**

| Model                         | Precision | Recall  | F1      |
| ----------------------------- | --------- | ------- | ------- |
| Bert-Base-NER             | **0.9124**    | **0.9189**  | **0.9157**  |
| Distillbert-cased fine-tuned (no mask) | 0.8927    | 0.9032  | 0.8979  |
| SpaCy Medium                  | 0.6618    | 0.5758  | 0.6158  |
| SpaCy Large                   | 0.6850    | 0.6317  | 0.6573  |
| Gemma 3 27B (zero-shot)       | 0.6361    | 0.7489  | 0.6879  |
| Gemma 3 27B (few-shot)        | 0.6689    | 0.7620  | 0.7125  |

**WikiAnn Results**

| Model                      | Precision | Recall  | F1      |
| -------------------------- | --------- | ------- | ------- |
| Bert-Base-NER              | **0.4786**    | **0.5287**  | **0.5024**  |
| Distillbert-cased fine-tuned (no mask) | 0.4603    | 0.5134  | 0.4854  |
| Spacy Medium               | 0.4002    | 0.3905  | 0.3953  |
| Spacy Large                | 0.4051    | 0.3987  | 0.4019  |