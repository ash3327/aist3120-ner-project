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