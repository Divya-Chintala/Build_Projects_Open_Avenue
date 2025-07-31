
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from collections import defaultdict
import os

def merge_entities_by_group(entities):
    grouped = defaultdict(list)
    for ent in entities:
        grouped[ent["entity_group"]].append(ent)
    merged = {}
    for label, ents in grouped.items():
        ents_sorted = sorted(ents, key=lambda e: e.get("start", 0))
        merged_text = " ".join(e["word"] for e in ents_sorted)
        merged[label] = (merged_text)
    return merged

def bert_prediction(sentence):
    base_path = os.path.dirname(os.path.dirname(__file__))  
    model_path = os.path.join(base_path, "trained_model")
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    ner_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="first"
    )

    results = ner_pipeline(sentence)


    merged = merge_entities_by_group(results)

    return merged