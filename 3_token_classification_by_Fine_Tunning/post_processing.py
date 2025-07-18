import json

def merge_exam_labels(dataset_path_in, dataset_path_out):
    with open(dataset_path_in, "r", encoding="utf-8") as f:
        data = json.load(f)

    for record in data:
        labels = record["labels"]

        merged_labels = []
        prev_label = "O"

        for label in labels:
            if label == "O":
                merged_labels.append("O")
            elif any(label.startswith(prefix) for prefix in ("B-Exam", "I-Exam", "E-Exam", "S-Exam",
                                                            "B-Technique", "I-Technique", "E-Technique", "S-Technique",
                                                            "B-Comparison", "I-Comparison", "E-Comparison", "S-Comparison")):
                # Map all these to "ExamName"
                # Replace BIOES prefix with same prefix but label "ExamName"
                bioes_prefix = label.split("-")[0]
                merged_labels.append(f"{bioes_prefix}-ExamName")
            else:
                # keep other labels as is
                merged_labels.append(label)

        record["labels"] = merged_labels

    with open(dataset_path_out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Saved merged labels dataset to {dataset_path_out}")


if __name__ == "__main__":
    merge_exam_labels("train.json", "train_merged.json")
    merge_exam_labels("test.json", "test_merged.json")
