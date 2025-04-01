# aist3120-ner-project: Named Entity Recognition

## Job Allocation

* Sam: Masked BERT.

## Progress

* 20250325: Load dataset (ongoing)

## Installation

* **Prerequisites**
    ```
    # *Open a new virtual environment before execution.
    pip install -r requirements.txt
    ```
* **Dataset: CoNLL-2003** | [Source](https://huggingface.co/datasets/eriktks) | [Data Description](https://huggingface.co/datasets/eriktks/conll2003#dataset-structure) | [Examples (for each NER class)](https://www.clips.uantwerpen.be/conll2003/ner/lists/)
* **Dataset: WikiAnn** | [Source](https://huggingface.co/datasets/unimelb-nlp/wikiann) | [Data Description](https://huggingface.co/datasets/unimelb-nlp/wikiann#dataset-structure)
  * Notes: Both dataset share the common set of NER tags from 0 to 6, but class 7 (B-MISC) and 8 (I-MISC) is not available in this dataset.
  * They both share the same format in `tokens` and `ner_tags` keys within each entry.

## Models
* **Baseline Provided:** Deep Contextualized Word Representations (2018) | [Paper](https://arxiv.org/pdf/1802.05365) | [PyTorch Implementation](https://github.com/yongyuwen/PyTorch-Elmo-BiLSTMCRF) | [TensorFlow Implementation](https://github.com/zhouyonglong/Deep-contextualized-word-representations-Tensorflow), [NER pre-trained model](https://github.com/allenai/allennlp-models)
* **Benchmark 1:** Spacy.
  * **Prerequisites:** `python -m spacy download <model_name>`, eg `en_core_web_md`.
* **Pre-trained model for NER** | [Source](https://huggingface.co/dslim/bert-large-NER)
  * This is fine-tuned on CoNLL-2003 dataset.
* `train_masked_bert.py`
  * v4: Fine-tune with the masked dataset for 3 epochs based on dslim/bert-base-NER.
  * v5: Fine-tune with the masked dataset for 9 epochs (same model as v4), with 1/3 the learning rate and lr decay. This is because 3 epochs may not allow the model to learn effectively from the masked dataset (due to random nature of the dataset).
  * nomask_v1 ([Reference](https://github.com/Louis-udm/NER-BERT-CRF)): 15 epochs, trained with raw dataset without masking.
  * 

## Papers Related

* Named Entity Recognition Using BERT with Whole
World Masking in Cybersecurity Domain (2021) | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9403180)

### Other References

* https://huggingface.co/docs/transformers/main/en/tasks/masked_language_modeling
* https://huggingface.co/docs/transformers/main/en/tasks/token_classification

## Metric

F-$\beta$ score, reduces to F-1 score when $\beta=1$:
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

<sup>#</sup>Hashtag: implies that this is a model that is already ready-to-use without fine-tuning or improvements.

| Model | Test Script |  CoNLL-2003 (train) | CoNLL-2003 (validation) | CoNLL-2003 (test) | WikiAnn (train) | WikiAnn (validation) | WikiAnn (test) |
| ---   | ---  | ---       | ---       | ---    | ---    |--  | --- |
| Spacy*<sup>#</sup> | `test_spacy_map.py` | | | 0.3849/0.5649/0.4578 |
| Spacy*<sup>#</sup> | `eval/eval_spacy.py` | 0.6932/0.6071/0.6473 | 0.7273/0.6390/0.6803 | 0.6618/0.5758/0.6158 | 0.4046/0.3972/0.4009 | 0.4192/0.4131/0.4161 | 0.4002/0.3905/0.3953 |
| Bert-Large-NER (dslim)<sup>#</sup> | `eval/eval_ftner_pretrained.py` | 0.8757/0.9013/0.8883 | 0.8310/0.8706/0.8503 | 0.8302/0.8637/0.8466 |  | 0.4241/0.5150/0.4652 | 0.4209/0.5083/0.4605
| Bert-FT-v1 | `masked_bert/masked_bert_ner.py --model_name runs/bert_ft_v1` (3 epochs, masked, fine-tuned on Bert-Base-Cased) | | | 0.7009/0.7397/0.7198 ||| 0.3153/0.3936/0.3501
| Bert-FT-v2 | `masked_bert/masked_bert_ner.py --model_name runs/bert_ft_v2` (5 epochs, masked) | | | 0.6741/0.7153/0.6941 ||| 0.3153/0.3936/0.3501
| Bert-FT-v3 | `masked_bert/masked_bert_ner.py --model_name runs/bert_ft_v3` (10 epochs, masked, evaluated with self-defined aggregation strategy (discards incomplete tokens)) | | | 0.6729/0.7151/0.6934 ||| 0.2947/0.3705/0.3283
| Bert-FT-v3 | `masked_bert/masked_bert_ner.py --model_name runs/bert_ft_v3` (evaluated with `aggregation_strategy="simple"`) | | | 0.6095/0.6972/0.6504 |
| Bert-FT-v3 | `masked_bert/masked_bert_ner.py --model_name runs/bert_ft_v3` (evaluated with `aggregation_strategy="first"`) | | | 0.8253/0.8693/0.8468 |
| Bert-FT-v3 | `masked_bert/masked_bert_ner.py --model_name runs/bert_ft_v3` (evaluated with `aggregation_strategy="max"`) | | | 0.8356/0.8497/0.8426 |
| Bert-Large-NER (dslim)<sup>#</sup> | `eval/eval_ftner_pretrained.py` (evaluated with `aggregation_strategy="max"`) ||| 0.8637/0.8867/0.8751 ||| 0.4163/0.5158/0.4607
| Bert-FT-v1 | `masked_bert/masked_bert_ner.py --model_name runs/bert_ft_v1` (evaluated with `aggregation_strategy="max"`) | | | 0.8425/0.8467/0.8446 |
| Bert-Base-NER (dslim)<sup>#</sup> | `masked_bert/masked_bert_ner.py --model_name dslim/bert-base-NER` (evaluated with `aggregation_strategy="max"`) |||0.8359/0.8817/0.8582|||0.4128/0.4943/0.4499
| Bert-FT-v4 | `masked_bert/masked_bert_ner.py --model_name runs/bert_ft_v4` (evaluated with `aggregation_strategy="max"`, fine-tund upon Bert-Base-NER (dslim)) | | | 0.8450/0.8860/0.8650 ||| 0.3869/0.4953/0.4344
| Bert-Base-Cased (dslim)<sup>#</sup> | `masked_bert/masked_bert_ner.py --model_name distilbert-base-cased` (evaluated with v3 aggregation strategy) |||SINCE IT IS NOT PRETRAINED OVER NER, THERE IS NO CLASSIFICATION RESULTS.
| Bert-Base-NER (dslim)<sup>#</sup> | `masked_bert/masked_bert_ner.py --model_name dslim/bert-base-NER` (evaluated with v3 aggregation strategy) |||0.9124/0.9189/0.9157|||0.4786/0.5287/0.5024
| Bert-FT-v1 | `masked_bert/masked_bert_ner.py --model_name runs/bert_ft_v1` (evaluated with v3 aggregation strategy, fine-tund upon Bert-Base-NER (dslim)) | | | 0.8929/0.89480.8939|||0.4577/0.5080/0.4816
| Bert-FT-v4 | `masked_bert/masked_bert_ner.py --model_name runs/bert_ft_v4` (evaluated with v3 aggregation strategy, fine-tund upon Bert-Base-NER (dslim)) | | | 0.9080/0.9171/0.9125 ||| 0.4583/0.5215/0.4879
| Bert-FT-v5 | `masked_bert/masked_bert_ner.py --model_name runs/bert_ft_v5` (evaluated with v3 aggregation strategy, fine-tund upon Bert-Base-NER (dslim)) | | | 0.9080/0.9170/0.9124 ||| 0.4572/0.5191/0.4862
| Bert-Base-NER (dslim)<sup>#</sup> | `masked_bert/evaluate_model.py --checkpoint_path dslim/bert-base-NER` (evaluated with seqeval) |||UNABLE TO PREDICT AS EXPECTED|||
| Bert-FT-v1 | `masked_bert/evaluate_model.py --checkpoint_path runs/bert_ft_v1` (evaluated with seqeval, fine-tund upon Bert-Base-NER (dslim)) | | | 0.8769/0.8945/0.8856||| 0.4057/0.4968/0.4466
| Bert-FT-v4 | `masked_bert/evaluate_model.py --checkpoint_path runs/bert_ft_v4` (evaluated with seqeval, fine-tund upon Bert-Base-NER (dslim)) | | | 0.9021/0.9186/0.9106||| 0.4257/0.5163/0.4666
| LLM (zero-shot) | `eval/eval_llm.py` model name: `gemma-3-27b-it` | | | 0.6361/0.7489/0.6879 | | | |
| LLM (few-shot) | `eval/eval_llm.py` model name: `gemma-3-27b-it` | | | 0.6689/0.7620/0.7125 | | | |
