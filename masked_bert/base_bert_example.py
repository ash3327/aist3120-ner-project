import sys
sys.path.append('.')

import numpy as np

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import pipeline
import evaluate
from string import punctuation

from eval.eval import Eval

puncts = punctuation

# Loading the models
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-cased")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

model = AutoModelForTokenClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=13, id2label=Eval.idx_to_label, label2id=Eval.label_to_idx
)

# Evaluation
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

training_args = TrainingArguments(
    output_dir="my_awesome_wnut_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_wnut["train"],
    eval_dataset=tokenized_wnut["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# # nlp = pipeline("ner", model=model, tokenizer=tokenizer)
# example = "My name is Wolfgang and I live in Berlin"

# # example = "Soccer - Japan get lucky win, China in surprise defeat."
# # example2 = ['SOCCER', '-', 'JAPAN', 'GET', 'LUCKY', 'WIN', ',', 'CHINA', 'IN', 'SURPRISE', 'DEFEAT', '.']
# example2 = ['L.', 'Germon', 'lbw', 'b', 'Afridi', '2']
# # example2 = ['I', 'thought', 'it', 'was', 'a', 'joke', ',', '"', 'said', 'Armando', 'who', 'replaces', 'injured', 'Atletico', 'Madrid', 'playmaker', 'Jose', 'Luis', 'Caminero', '.']
# example = ' '.join(example2)#.lower()
# for punct in puncts:
#     example = example.replace(' '+punct, punct)
# print(example)

# def parse(example):
#     tokens = tokenizer(example, return_tensors="pt")
#     outputs = model(**tokens)
#     ner_results = outputs.logits
#     print(ner_results)
#     # token_ids = tokenizer(example)['input_ids']
#     # tokens = tokenizer.convert_ids_to_tokens(token_ids)
#     # print(tokens)
#     return ner_results
#     # if len(ner_results) == 0:
#     #     return []
#     # out = list()
#     # lastid = ner_results[0]['index']
#     # lastitem = ner_results[0]['word']
#     # lastlabel = ner_results[0]['entity'][2:]
#     # for result in ner_results[1:]:
#     #     if result['index'] == lastid+1:
#     #         if result['word'].startswith('##'):
#     #             lastitem += result['word'][2:]
#     #         elif result['word'] in puncts:
#     #             lastitem += result['word']
#     #         else:
#     #             lastitem += ' ' + result['word']
#     #         lastid += 1
#     #     else:
#     #         if lastitem:
#     #             out.append((lastlabel, lastitem))
#     #         if result['word'].startswith('##'):
#     #             lastid = -1
#     #             lastitem = None
#     #             lastlabel = None
#     #         else:
#     #             lastid = result['index']
#     #             lastitem = result['word']
#     #             lastlabel = result['entity'][2:]
#     # out.append((lastlabel, lastitem))

#     # print(out)
#     # # ner_results = list(zip(*[[result['word'], result['entity'], result['score'], result['index']] for result in ner_results]))
#     # # print(ner_results)
#     # return out

#     # print(tokenizer(example))
#     # print(model(tokenizer(example)))

# print(parse(example))