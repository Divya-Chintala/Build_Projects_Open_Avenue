import os
import json
import re
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from difflib import SequenceMatcher

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', text.strip())

def find_span_start(text, value, use_fuzzy=True, threshold=0.9):
    """Find exact or fuzzy match start index of value in text."""
    start = text.find(value)
    if start != -1:
        return start

    if not use_fuzzy:
        return -1

    # Fuzzy match fallback
    best_ratio = 0
    best_start = -1
    window_size = len(value) + 10
    for i in range(len(text) - len(value)):
        window = text[i:i + window_size]
        ratio = SequenceMatcher(None, window, value).ratio()
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_start = i

    return best_start


def get_bioes_labels(text, fields):
    text = normalize_whitespace(text)
    tokens = tokenizer.tokenize(text)
    token_offsets = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    labels = ["O"] * len(tokens)

    for field, value in fields.items():
        value = normalize_whitespace(value)
        if not value:
            continue

        start = find_span_start(text, value)
        if start == -1:
            print(f" Span not found for field: {field}")
            print(f"Value: {repr(value)}")
            print(f"Text sample: {repr(text[:200])}...\n")
            continue

        end = start + len(value)

        # Find token indices within span
        span_indices = [i for i, (s, e) in enumerate(token_offsets) if s >= start and e <= end]
        if not span_indices:
            continue

        if len(span_indices) == 1:
            labels[span_indices[0]] = f"S-{field}"
        else:
            labels[span_indices[0]] = f"B-{field}"
            for i in span_indices[1:-1]:
                labels[i] = f"I-{field}"
            labels[span_indices[-1]] = f"E-{field}"

    return {"tokens": tokens, "labels": labels}


def extract_exam_components(raw_exam_field):
    """Extracts Exam, Technique, and Comparison sections from ExamName using regex."""
    sections = {
        "Exam": "",
        "Technique": "",
        "Comparison": ""
    }

    if not raw_exam_field:
        return sections

    # Normalize line breaks and spaces
    raw_exam_field = normalize_whitespace(raw_exam_field)

    exam_match = re.search(r"(EXAM:.*?)(?=TECHNIQUE:|COMPARISON:|$)", raw_exam_field, re.IGNORECASE)
    tech_match = re.search(r"(TECHNIQUE:.*?)(?=EXAM:|COMPARISON:|$)", raw_exam_field, re.IGNORECASE)
    comp_match = re.search(r"(COMPARISON:.*)", raw_exam_field, re.IGNORECASE)

    if exam_match:
        sections["Exam"] = exam_match.group().strip()
    if tech_match:
        sections["Technique"] = tech_match.group().strip()
    if comp_match:
        sections["Comparison"] = comp_match.group().strip()

    return sections


def process_json_directory(input_dir):
    data = []
    for fname in tqdm(os.listdir(input_dir)):
        if not fname.endswith(".json"):
            continue

        with open(os.path.join(input_dir, fname), "r", encoding="utf-8") as f:
            record = json.load(f)

        report_text = record.get("ReportText", "")
        raw_exam_field = record.get("ExamName", "")
        exam_parts = extract_exam_components(raw_exam_field)

        fields = {
            "Exam": exam_parts["Exam"],
            "Technique": exam_parts["Technique"],
            "Comparison": exam_parts["Comparison"],
            "clinicaldata": record.get("clinicaldata", ""),
            "findings": record.get("findings", ""),
            "impression": record.get("impression", "")
        }

        labeled = get_bioes_labels(report_text, fields)

        if any(label.startswith("B-") for label in labeled["labels"]):
            data.append(labeled)

    # Train-test split
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    with open("train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2)
    with open("test.json", "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2)

    print(f"Saved {len(train)} training and {len(test)} test examples.")


if __name__ == "__main__":
    process_json_directory(
        input_dir=r"C:\Users\divya\Documents\GitHub\Build_Projects_Open_Avenue\3_token_classification_by_Fine_Tunning\openave_jsons"
    )
