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

## Metric

F-$\beta$ score, reduces to F-1 score when $\beta=1$:
$$F_\beta=\frac{(\beta^2+1)\times\text{precision}\times\text{recall}}{\beta^2\times\text{precision}+\text{recall}}$$

where, precision and recalls are evaluated on whether the named entity is being identified (and matches exactly as the one in the data file).

## Results (*raw)

Metric: Precision/Recall/F1 Score

<sup>#</sup>Hashtag: implies that this is a model that is already ready-to-use without fine-tuning or improvements.

| Model | Test Script |  CoNLL-2003 (train) | CoNLL-2003 (test) | WikiAnn (train) | WikiAnn (test) |
| ---   | ---         | ---       | ---    | ---      | --- |
| Spacy*<sup>#</sup> | `test_spacy_map.py` | | 0.3849/0.5649/0.4578 |
| Spacy*<sup>#</sup> | `eval/eval_spacy.py` | 0.6932/0.6071/0.6473 | 0.6618/0.5758/0.6158 | 0.4046/0.3972/0.4009 | 0.4002/0.3905/0.3953 |
| Bert-Large-NER (dslim)<sup>#</sup> | `eval/eval_ftner_pretrained.py` | 0.8757/0.9013/0.8883 | 0.8302/0.8637/0.8466 | 0.4209/0.5083/0.4605
