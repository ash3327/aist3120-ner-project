import spacy

# Print explanations for each entity label
print("\nEntity Label Explanations:")
print("--------------------------")
entity_labels = ["CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", 
                "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT", 
                "QUANTITY", "TIME", "WORK_OF_ART"]

for label in entity_labels:
    explanation = spacy.explain(label)
    print(f"{label}: {explanation}")
