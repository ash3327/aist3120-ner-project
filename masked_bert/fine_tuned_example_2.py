from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from string import punctuation

puncts = punctuation

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
example = "My name is Wolfgang and I live in Berlin"

# example = "Soccer - Japan get lucky win, China in surprise defeat."
# example2 = ['SOCCER', '-', 'JAPAN', 'GET', 'LUCKY', 'WIN', ',', 'CHINA', 'IN', 'SURPRISE', 'DEFEAT', '.']
example2 = ['L.', 'Germon', 'lbw', 'b', 'Afridi', '2']
# example2 = ['I', 'thought', 'it', 'was', 'a', 'joke', ',', '"', 'said', 'Armando', 'who', 'replaces', 'injured', 'Atletico', 'Madrid', 'playmaker', 'Jose', 'Luis', 'Caminero', '.']
example = ' '.join(example2)#.lower()
for punct in puncts:
    example = example.replace(' '+punct, punct)
print(example)

def parse(example):
    ner_results = nlp(example)
    print(ner_results)
    # token_ids = tokenizer(example)['input_ids']
    # tokens = tokenizer.convert_ids_to_tokens(token_ids)
    # print(tokens)
    return
    if len(ner_results) == 0:
        return []
    out = list()
    lastid = ner_results[0]['index']
    lastitem = ner_results[0]['word']
    lastlabel = ner_results[0]['entity'][2:]
    for result in ner_results[1:]:
        if result['index'] == lastid+1:
            if result['word'].startswith('##'):
                lastitem += result['word'][2:]
            elif result['word'] in puncts:
                lastitem += result['word']
            else:
                lastitem += ' ' + result['word']
            lastid += 1
        else:
            if lastitem:
                out.append((lastlabel, lastitem))
            if result['word'].startswith('##'):
                lastid = -1
                lastitem = None
                lastlabel = None
            else:
                lastid = result['index']
                lastitem = result['word']
                lastlabel = result['entity'][2:]
    out.append((lastlabel, lastitem))

    print(out)
    # ner_results = list(zip(*[[result['word'], result['entity'], result['score'], result['index']] for result in ner_results]))
    # print(ner_results)
    return out

    # print(tokenizer(example))
    # print(model(tokenizer(example)))

print(parse(example))